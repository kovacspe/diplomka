import MEI_network
import Sinz2018_NIPS

train,val,test = DataLoader.get_data()
net = MEI_network.define_MEI_network()