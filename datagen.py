import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

ct_dir ='BLD/CT_vol/'
seg_dir = 'BLD/Seg/GT/'
dist_xfm_dir = 'BLD/SDT/GT/'

source_dir = "/path/to/bony_labyrinth_dataset/" #F01/F01/CT/"    ## TODO: update path variable here ##
dataset_dir_uCT = "BLD/DS_uCT/"
dataset_dir_CT = "BLD/DS_CT/"

specimen_list=os.listdir(source_dir)
# print(specimen_list)

for i in tqdm(range(len(specimen_list))):    
    if i<15: 
        filename = str(dataset_dir_uCT) + str("train/") + f"{specimen_list[i]}.h5" 
        with h5py.File(filename,'w') as data:
            data.create_dataset('input',data = np.load(os.path.join(ct_dir,f"{specimen_list[i]}.npy")))
            data.create_dataset('target_seg',data = np.load(os.path.join(seg_dir,f"{specimen_list[i]}.npy"))) 
            data.create_dataset('target_sdt',data = np.load(os.path.join(dist_xfm_dir,f"{specimen_list[i]}.npy")))
    elif i>15 and i<20:
        filename = str(dataset_dir_uCT) + str("val/") + f"{specimen_list[i]}.h5" 
        with h5py.File(filename,'w') as data:
            data.create_dataset('input',data = np.load(os.path.join(ct_dir,f"{specimen_list[i]}.npy")))
            data.create_dataset('target_seg',data = np.load(os.path.join(seg_dir,f"{specimen_list[i]}.npy"))) 
            data.create_dataset('target_sdt',data = np.load(os.path.join(dist_xfm_dir,f"{specimen_list[i]}.npy")))
    else:
        filename = str(dataset_dir_uCT) + str("test/") + f"{specimen_list[i]}.h5" 
        with h5py.File(filename,'w') as data:
            data.create_dataset('input',data = np.load(os.path.join(ct_dir,f"{specimen_list[i]}.npy")))
            data.create_dataset('target_seg',data = np.load(os.path.join(seg_dir,f"{specimen_list[i]}.npy"))) 
            data.create_dataset('target_sdt',data = np.load(os.path.join(dist_xfm_dir,f"{specimen_list[i]}.npy")))


