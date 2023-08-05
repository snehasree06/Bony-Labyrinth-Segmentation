import pathlib
import random
import numpy as np
import h5py
from torch.utils.data import Dataset
import torch
from skimage import feature
import os 

class VolData(Dataset):
    """
    A PyTorch Dataset that provides access to CT Volumes.
    """
    def __init__(self, root): 
        # List the h5 files in root 
        files = list(pathlib.Path(root).iterdir())
        self.examples = []
        self.key_img = 'input'

        for fname in sorted(files):
            with h5py.File(fname,'r') as hf:
                #print (hf.keys())
                fsvol = hf['target_seg']
                num_slices = fsvol.shape[2]
                self.examples += [(fname, slice) for slice in range(num_slices)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i] 
        with h5py.File(fname, 'r') as data:
            input_img  = data[self.key_img][:,:,slice].astype(np.float64)
            target = data['target_seg'][:,:,slice].astype(np.float64)
            # print(f"input_shape:{input_img.shape},target_shape:{target.shape}")
            # print(f"min:{np.min(input_img)},max:{np.max(input_img)}")
            # quit()
            return torch.from_numpy(input_img), torch.from_numpy(target)
        

