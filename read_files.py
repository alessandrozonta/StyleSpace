import numpy as np
import pickle


their = "npy/ffhq/W.npy"
dlatents=np.load(their)

mine = "npy/ffhq/W_mine.npy"
dlatents_mine=np.load(mine)


s = "npy/ffhq/S"
with open(s, "rb") as fp:   #Pickling
    s_names,all_s=pickle.load( fp)

print("a")