from data_loaders import DataLoader, get_data_loader
from utils import reshape_input_to_NDN,merge_train_and_val_set, evaluate_performance
from antolik_models import FCModel, ConvDoGModel, DoGModel, SimpleConvModel
import fire

class Trainer:
    def __init__(self,data_loader,model,kwargs):
        self.data = data_loader
        self.model = model
        self.args = kwargs


    def print_evalution(self,net):
        train_x, train_y = self.data.train(NDN_reshape=True)
        pred = net.generate_prediction(train_x)
        corr = evaluate_performance(pred, train_y)
        print('Train correlation :', corr)

        val_x, val_y = self.data.val(NDN_reshape=True)
        pred = net.generate_prediction(val_x)
        corr = evaluate_performance(pred, val_y)
        print('Train correlation :', corr)

        test_x, test_y = self.data.train(NDN_reshape=True)
        if test_x:
            pred = net.generate_prediction(test_x)
            corr = evaluate_performance(pred, test_y)
            print('Test correlation :', corr)


    def run_experiment(self,experiment_name,save_name=None):
        train_x, train_y = self.data.train(NDN_reshape=True)
        val_x, val_y = self.data.val(NDN_reshape=True)
        data_x,data_y,train_idxs,test_idxs = merge_train_and_val_set(train_x,train_y,val_x,val_y)

        name = f'{experiment_name}-{self.model.get_name()}'

        self.net = self.model.get_net()
        opt_params = self.model.get_opt_params()
        if 'gpu' in self.args:
            opt_params['use_gpu'] = True
        self.net.log_correlation = 'filter-low-std-gold'
        self.net.train(
            data_x,
            data_y,
            train_indxs=train_idxs,
            test_indxs=test_idxs,
            learning_alg='adam',
            opt_params=opt_params,
            output_dir=f'output/{name}'
        )
        self.print_evaluation(self.net)
        if save_name:
            self.net.save_model(f'models/{name}.pkl')

models = {
    #'iclr': ICLRModel,
    'base': FCModel,
    'conv': SimpleConvModel,
    'dog': DoGModel,
    'convdog': ConvDoGModel
}

def run(model_type,data_type,**kwargs):
    print('Trainer kwargs: ',kwargs)
    model_class = models[model_type]
    print('Loading data')
    data_loader = get_data_loader(data_type)
    print('Constructing model...')
    model = model_class(data_loader,kwargs)
    print('creating_trainer')
    trainer = Trainer(data_loader,model,kwargs)
    print('Running_experiment')
    trainer.run_experiment('exp')
    

if __name__ == '__main__':
    fire.Fire(run)
    