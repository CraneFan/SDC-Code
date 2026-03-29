import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
from torch.utils.data import DataLoader

def train(args):

    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    

    args["device"] = device
    _train(args, seed_list)

def _calculate_helmholtz_free_energy(model, data_loader, temperature=1.0):

    model._network.eval()
    total_free_energy = 0.0
    total_samples = 0
    device = model._device

    with torch.no_grad():
        for _, inputs, _ in data_loader:
            inputs = inputs.to(device)

            outputs = model._network(inputs)
            logits = outputs['logits']


            free_energy_per_sample = -torch.logsumexp(logits, dim=1)

            total_free_energy += torch.sum(free_energy_per_sample).item()
            total_samples += len(inputs)
            

    return total_free_energy / total_samples if total_samples > 0 else float('inf')


def _train(args, seed_list):

    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    if not os.path.exists(logs_name): os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}".format(args["model_name"],args["dataset"],init_cls,args["increment"],args["prefix"],args["backbone_type"])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(filename)s] => %(message)s", handlers=[logging.FileHandler(filename=logfilename + ".log"), logging.StreamHandler(sys.stdout)])
    
    _set_device(args)
    print_args(args)
    data_manager = DataManager(args["dataset"], args["shuffle"], args["seed"][0], args["init_cls"], args["increment"], args)
    args["nb_classes"] = data_manager.nb_classes
    args["nb_tasks"] = data_manager.nb_tasks


    num_models = len(seed_list)
    logging.info(f"Initializing a model pool with {num_models} models, seeds: {seed_list}")
    model_pool = []
    for seed in seed_list:
        logging.info(f"Creating model with seed: {seed}")

        args_copy = copy.deepcopy(args)
        args_copy["seed"] = seed
        _set_random(seed)

        model = factory.get_model(args_copy["model_name"], args_copy)


        model._network.to(args["device"])
        model_pool.append(model)
        
    task_counts = np.zeros(num_models, dtype=int)
    max_tasks_per_model = (args["nb_tasks"] // num_models) + 1
    logging.info(f"Maximum number of tasks per model: {max_tasks_per_model}")


    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    for task_id in range(data_manager.nb_tasks):
        logging.info(f"================== Task {task_id+1} ==================")
        

        best_model_idx = -1
        min_free_energy = float('inf')


        fitness_dataset = data_manager.get_dataset(np.arange(data_manager.get_task_size(task_id) * task_id, data_manager.get_task_size(task_id) * (task_id+1)), source="train", mode="test")

        num_workers = args.get("num_workers", 0)
        fitness_loader = DataLoader(fitness_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=num_workers)

        logging.info(f"Calculating Helmholtz free energy for available models...")
        for i, model in enumerate(model_pool):

            if task_counts[i] < max_tasks_per_model:

                free_energy = _calculate_helmholtz_free_energy(model, fitness_loader)
                logging.info(f"  - Model {i} (Seed: {seed_list[i]}, Tasks trained: {task_counts[i]}) -> Free Energy: {free_energy:.4f}")
                if free_energy < min_free_energy:
                    min_free_energy = free_energy
                    best_model_idx = i
            else:
                logging.info(f"  - Model {i} (Seed: {seed_list[i]}) has reached its task limit of {max_tasks_per_model}.")

        if best_model_idx == -1:
            logging.error("No available model to train the current task. Exiting.")
            break
            
        selected_model = model_pool[best_model_idx]
        logging.info(f"Task {task_id + 1} assigned to Model {best_model_idx} (Seed: {seed_list[best_model_idx]}) with the lowest free energy.")


        logging.info(f"Training Model {best_model_idx} on Task {task_id + 1}...")
        selected_model.incremental_train(data_manager)
        

        task_counts[best_model_idx] += 1
        

        cnn_accy, nme_accy = selected_model.eval_task()
        selected_model.after_task()
        

        if nme_accy is not None:
            cnn_curve["top1"].append(cnn_accy["top1"])
            nme_curve["top1"].append(nme_accy["top1"])
            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
        else:
            cnn_curve["top1"].append(cnn_accy["top1"])
            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

    logging.info("Final task assignments per model:")
    for i in range(num_models):
        logging.info(f"  - Model {i} (Seed: {seed_list[i]}) trained {task_counts[i]} tasks.")



def _set_device(args):
    device_type = args["device"]

    if isinstance(device_type, torch.device):

        gpus = [device_type]
    elif isinstance(device_type, (list, tuple)):

        gpus = []
        for device in device_type:
            if str(device) == "-1":
                device = torch.device("cpu")
            else:
                device = torch.device("cuda:{}".format(device))
            gpus.append(device)
    else:

        if str(device_type) == "-1":
            gpus = [torch.device("cpu")]
        else:
            gpus = [torch.device(f"cuda:{device_type}") if str(device_type).isdigit() else torch.device(device_type)]
    args["device"] = gpus[0]

def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))