import numpy as np

class AntolikDataLoader:
    def __init__(self, data_folder):
        self.data_folder=data_folder

    def load_train(self,region):
        return self.load(region,'training')

    def load_val(self,region):
        return self.load(region,'validation')

    def load(self,region,data_type):
        x = np.load(f'{self.data_folder}/region{region}/{data_type}_inputs.npy')
        y = np.load(f'{self.data_folder}/region{region}/{data_type}_set.npy')
        return x,y

if __name__ == '__main__':
    dataset = AntolikDataLoader('antolik_data')
    data_in,data_out = dataset.load_train(1)
    print(data_in.shape)
    print(data_out.shape)
