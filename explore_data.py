import matplotlib.pyplot as plt
import numpy as np


from utils.data_loaders import AntolikDataLoader,ICLRDataLoader
import fire

def explore_rotaion_dataset_data():
    data = ICLRDataLoader('data.pkl')
    x,y = data.train()
    print(x.shape)
    print(f'first pixel mean: {np.mean(x[:,0,0,0])} std: {np.std(x[:,0,0,0])}')
    print(f'first picture mean: {np.mean(x[0,:,:,0])} std: {np.std(x[0,:,:,0])}')
    means = []
    stds = []
    for i in range(x.shape[1]):
        for j in range(x.shape[2]):
            means.append(np.mean(x[:,i,j,0]))
            stds.append(np.std(x[:,i,j,0]))

    means = np.array(means)
    stds = np.array(stds)
    print(f'Means: {np.mean(means)}, stds:{np.mean(stds)}')
    a = input()
    #output exporation
    print(np.mean(y[:,0]))
    means = []
    stds = []
    for i in range(y.shape[1]):
        means.append(np.mean(y[:,i]))
        stds.append(np.std(y[:,i]))
    #plt.boxplot(means)
    #plt.show()

    means = np.array(means)
    print(f'y shape:{y.shape}')
    print('Means: ',np.mean(means))
    stds = np.array(stds)
    print('Väčšia std: ',np.sum(np.where(stds>1,1,0)))
    print('Malá std: ',np.sum(np.where(stds<=1e-5,1,0)))
    #for i in range(15):
    #    plt.figure()
    #    plt.imshow(x[i,:,:,0])
    #plt.show()

def explore_antolik_data(region=1,data_path='data'):
    dataset = AntolikDataLoader(data_path,region)
    sample_images, sample_activations = dataset.train(NDN_reshape=True)
    val_x,val_y = dataset.val()

    print(f'Number of samples:')
    print(f'Train: {len(sample_images)}')
    print(f'Test: {len(val_x)}')
    x,y = dataset.width ,dataset.height
    print(f'Image shape: {x}x{y} pixels')
    print(f'Number of neurons: {dataset.num_neurons}')
    print('Image statistics')
    print(f'Min: {np.min(sample_images)}')
    print(f'Max: {np.max(sample_images)}')
    print(f'Mean: {np.mean(sample_images)}')
    print(f'Std: {np.std(sample_images)}')
    print(f'Mean L2 of image: {np.mean(np.sum(sample_images**2,axis=1))}')
    sample_images = np.reshape(sample_images,(-1,x,y))
    fig, ax1 = plt.subplots(3, 4,figsize=(40,25))
    for i, res in enumerate(sample_images[:12]):
        ax1[i % 3, (i//3)].imshow(res, cmap='gray')
        ax1[i % 3, (i//3)].get_xaxis().set_visible(False)
        ax1[i % 3, (i//3)].get_yaxis().set_visible(False)
    plt.savefig(f'output/01_data_exploration/region{region}.png')

if __name__=="__main__":
    fire.Fire()