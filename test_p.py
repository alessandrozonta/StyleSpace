import pickle

with open("npy/my_out/S_1", "rb") as fp:   #Pickling
    s_names,all_s=pickle.load( fp)
dlatents=all_s