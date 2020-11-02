import NDN3.NDN as NDN

class Model:
    def __init__(self,data_loader,args):
        self.data_loader = data_loader
        self.width = data_loader.width
        self.height = data_loader.height
        self.out_num = data_loader.num_neurons
        self.args = args
        self.net_name='generic'
        self.opt_params = {}

    def get_params(self):
        return {}

    def get_net(self):
        params = self.get_params()
        print(self.opt_params)
        bs = self.get_opt_params()['batch_size']
        seed = self.get_opt_params()['seed'] if 'seed' in self.get_opt_params() else 0
        net = NDN.NDN(params,
                      input_dim_list=[[1, self.data_loader.height, self.data_loader.width]],
                      batch_size=bs,
                      noise_dist='poisson',
                      tf_seed=seed)
        return net

    def get_opt_params(self):
        return self.opt_params

    def get_name(self):
        name = self.net_name
        for key,value in self.args.items():
            name+=f'-{key}{value}'
        return name

        