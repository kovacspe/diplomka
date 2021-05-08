import yaml
from NDN3.NDN import NDN
from utils.data_loaders import get_data_loader

def experiment_args(func):
    def wrapper(*args,**kwargs):
        # Load configuration if experiment is specified
        update_params = {}
        if 'experiment' in kwargs:
            exp_id = kwargs['experiment']
            with open('utils/experiments.yml','r') as conf_file:
                loaded_yaml = yaml.load(conf_file)
            loaded_params = loaded_yaml.get(exp_id,{})
            if 'model_exp_ids' in loaded_params:
                if isinstance(loaded_params['model_exp_ids'],list):
                    pass
            # Filter configuration params and update by **kwargs
            varnames = func.__code__.co_varnames
            for key,value in loaded_params.items():
                if key in varnames:
                    update_params[key] = value
        update_params.update(kwargs)
        print(update_params)
        kwargs = update_params
        # Load model if net path in args
        if 'net' in kwargs and isinstance(kwargs['net'],str):
            kwargs['net'] = NDN.load_model(kwargs['net'])

        # Load dataset
        if 'dataset' in kwargs and isinstance(kwargs['dataset'],str):
            kwargs['dataset'] = get_data_loader(kwargs['dataset'])

        return func(**kwargs)
    return wrapper