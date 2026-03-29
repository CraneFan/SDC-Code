# utils/factory.py

def get_model(model_name, args):
    name = model_name.lower()


    if name == "sdc":
        from models.SDC import Learner
        return Learner(args)
    if name == "sdlora":
        from models.SDC import Learner
        return Learner(args)
    elif name == "simplecil":
        from models.simplecil import Learner
        return Learner(args)
    elif name == "adam_finetune":
        from models.adam_finetune import Learner
        return Learner(args)
    elif name == "adam_ssf":
        from models.adam_ssf import Learner
        return Learner(args)
    elif name == "adam_vpt":
        from models.adam_vpt import Learner 
        return Learner(args)
    elif name == "adam_adapter":
        from models.adam_adapter import Learner
        return Learner(args)
    elif name == "l2p":
        from models.l2p import Learner
        return Learner(args)
    elif name == "dualprompt":
        from models.dualprompt import Learner
        return Learner(args)
    elif name == "coda_prompt":
        from models.coda_prompt import Learner
        return Learner(args)
    elif name == "finetune":
        from models.finetune import Learner
        return Learner(args)
    elif name == "icarl":
        from models.icarl import Learner
        return Learner(args)
    elif name == "der":
        from models.der import Learner
        return Learner(args)
    elif name == "coil":
        from models.coil import Learner
        return Learner(args)
    elif name == "foster":
        from models.foster import Learner
        return Learner(args)
    elif name == "memo":
        from models.memo import Learner
        return Learner(args)  
    else:
        assert 0, "Unknown model name: {}".format(model_name)
