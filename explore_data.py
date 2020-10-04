import matplotlib.pyplot as plt
import numpy as np

from load_rotation_data import get_data, Dataset

data = get_data('data.pkl')

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
