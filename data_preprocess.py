import os
from os.path import join
import numpy as np
import SimpleITK as sitk


from tqdm import tqdm

import torch
from skimage.transform import rescale, resize
from scipy.ndimage import distance_transform_edt
from scipy.signal import fftconvolve
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import directed_hausdorff
import h5py

device = "cuda" if torch.cuda.is_available() else "cpu" #cuda is for gpu

source_dir = "/path/to/bony_labyrinth_dataset/" #F01/F01/CT/"    ## TODO: update path variable here ##
output_dir ='BLD/'

dataset_dir_uCT = "BLD/DS_uCT/"
# os.listdir(source_dir)

specimen_list=os.listdir(source_dir)
# specimen_list = [fname for fname in os.listdir(source_dir) if fname!="F12"] #CT

pat_dir_uCT_RAW=[]
for i in range(0,len(specimen_list)):
    pat_dir_uCT_RAW.append(source_dir+specimen_list[i]+"/"+specimen_list[i]+"/uCT/"+f"{specimen_list[i]}_uCT_RAW.nii")

pat_dir_uCT_LABELS=[]
for i in range(0,len(specimen_list)):
    pat_dir_uCT_LABELS.append(source_dir+specimen_list[i]+"/"+specimen_list[i]+"/uCT/"+f"{specimen_list[i]}_uCT_LABELS.nii")

def Normalize(image):
    minval = image.min()
    maxval = image.max()
    wld = np.clip(image, minval, maxval)
    wld -= minval
    wld *= (1 / (maxval-minval))
    return wld

for i in tqdm(range(len(specimen_list))):
    
    CT = sitk.ReadImage(pat_dir_uCT_RAW[i])
    CT_spacing = np.array(CT.GetSpacing())
    npy_CT = sitk.GetArrayFromImage(CT)
    
    seg = sitk.ReadImage(pat_dir_uCT_LABELS[i])
    seg_spacing = np.array(seg.GetSpacing())
    npy_seg = sitk.GetArrayFromImage(seg)  
    
    new_shape = (256,256,256)
    npy_CT = resize(npy_CT, new_shape, order=0, mode='constant', cval=0, clip=True, preserve_range=True)
    npy_CT = Normalize(npy_CT.astype(float))
    npy_seg = resize(npy_seg, new_shape, order=0, mode='constant', cval=0, clip=True, preserve_range=True)
    
    np.save(join(output_dir, "CT_vol",f"{specimen_list[i]}.npy"), npy_CT)
    np.save(join(output_dir, "Seg","GT", f"{specimen_list[i]}.npy"), npy_seg)
    
    assert((npy_seg.shape==npy_CT.shape))
    num_deformations_to_generate_per_seg = 100
    desired_spacing = CT_spacing
    # get distance xfm
    dist_xfm = distance_transform_edt((~(npy_seg.astype(bool))).astype(float), sampling=desired_spacing) - distance_transform_edt(npy_seg, sampling=desired_spacing)    
    np.save(join(output_dir, "SDT", "GT",f"{specimen_list[i]}.npy"), dist_xfm)
    
        
    
