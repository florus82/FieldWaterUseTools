# some of the functions listed here were taken from https://github.com/feevos/tfcl/tree/master
# this repo is connected to the AI4boudnaries dataset https://doi.org/10.5194/essd-15-317-2023

# load functions and packages
import random
import torch
import pickle
import time
from tqdm import tqdm
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import albumentations as A
from albumentations.core.transforms_interface import  ImageOnlyTransform
from torch.amp import autocast
from skimage import measure
import higra as hg
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from osgeo import gdal
gdal.DontUseExceptions()
from tfcl.utils.classification_metric import Classification
from tfcl.models.ptavit3d.ptavit3d_dn import ptavit3d_dn
from helperFuncs import *


def shuffle2Lists(list1, list2):
   
    paired = list(zip(list1, list2))
    random.shuffle(paired)
    list1_1, list2_2 = zip(*paired)

    list1 = list(list1_1)
    list2 = list(list2_2)

    return list1, list2 


class TrainingTransformS2(object):
    # Built on Albumentations, this provides geometric transformation only  
    def __init__(self,  prob = 1., mode='train', norm = AI4BNormal_S2() ):
        self.geom_trans = A.Compose([
                    #A.RandomCrop(width=128, height=128, p=1.0),  # Always apply random crop
                    A.OneOf([
                        A.HorizontalFlip(p=1),
                        A.VerticalFlip(p=1),
                        A.ElasticTransform(p=1), # VERY GOOD - gives perspective projection, really nice and useful - VERY SLOW   
                        A.GridDistortion(distort_limit=0.4,p=1.),
                        A.ShiftScaleRotate(shift_limit=0.25, scale_limit=(0.75,1.25), rotate_limit=180, p=1.0), # Most important Augmentation   
                        ],p=1.)
                    ],
            additional_targets={'imageS1': 'image','mask':'mask'},
            p = prob)
        if mode=='train':
            self.mytransform = self.transform_train
        elif mode =='valid':
            self.mytransform = self.transform_valid
        else:
            raise ValueError('transform mode can only be train or valid')
            
        self.norm = norm
        
    def transform_valid(self, data):
        timgS2, tmask = data
        if self.norm is not None:
            timgS2 = self.norm(timgS2)
        
        tmask= tmask 
        return timgS2,  tmask.astype(np.float32)

    def transform_train(self, data):
        timgS2, tmask = data
        
        if self.norm is not None:
            timgS2 = self.norm(timgS2)

        tmask= tmask 
        tmask = tmask.astype(np.float32)
        # Special treatment of time series
        c2,t,h,w = timgS2.shape
        #print (c2,t,h,w)              
        timgS2 = timgS2.reshape(c2*t,h,w)
        result = self.geom_trans(image=timgS2.transpose([1,2,0]),
                                 mask=tmask.transpose([1,2,0]))
        timgS2_t = result['image']
        tmask_t  = result['mask']
        timgS2_t = timgS2_t.transpose([2,0,1])
        tmask_t = tmask_t.transpose([2,0,1])
        
        c2t,h2,w2 = timgS2_t.shape

        timgS2_t = timgS2_t.reshape(c2,t,h2,w2)
        return timgS2_t,  tmask_t
    
    def __call__(self, *data):
        return self.mytransform(data)

    
class AI4BDataset(torch.utils.data.Dataset):
    def __init__(self, list_of_imgs, list_of__masks, transform=TrainingTransformS2(), mode='train', ntrain=0.9): #path_to_data=r'/path/to/AI4BOUNDARIES/sentinel2/'
        
        self.flnames_s2_img = list_of_imgs 
        self.flnames_s2_mask = list_of__masks

        assert len(self.flnames_s2_img) == len(self.flnames_s2_mask), ValueError("Some problem, the masks and images are not in the same numbers, aborting")
        
        tlen = len(self.flnames_s2_img)
        
        if mode=='train':
            self.flnames_s2_img = self.flnames_s2_img[:int(ntrain*tlen)]
            self.flnames_s2_mask = self.flnames_s2_mask[:int(ntrain*tlen)]
        elif mode=='valid':
            self.flnames_s2_img = self.flnames_s2_img[int(ntrain*tlen):]
            self.flnames_s2_mask = self.flnames_s2_mask[int(ntrain*tlen):]
        else:
            raise ValueError("Cannot undertand mode::{}, should be either train or valid, aborting...".format(mode))
        

        self.transform=transform                                                              
    
    # Helper function to read nc to raster 
    def ds2rstr(self,tname):

        variables2use=["Blue", "Green", "Red", "Infra"]
        ds = xr.open_dataset(tname)
        ds_np = np.concatenate([ds[var].values[None] for var in variables2use],0)

        return ds_np

    def read_mask(self,tname):
        return rasterio.open(tname).read((1,2,3))

    
    def __getitem__(self,idx):

        tname_img = self.flnames_s2_img[idx]
        tname_mask = self.flnames_s2_mask[idx]
        
        timg = self.ds2rstr(tname_img)
        tmask = self.read_mask(tname_mask)
        
        if self.transform is not None:
            timg, tmask = self.transform(timg,tmask)
            
        return timg, tmask
    
    def __len__(self):
        return len(self.flnames_s2_img)


class AI4BPatchDataset(AI4BDataset):
    def __init__(self, list_of_imgs, list_of__masks, patch_size=128, stride=64, transform=None, mode='train', ntrain=0.9): # path_to_data
        """
        path_to_data: root folder with images and masks
        patch_size: size of extracted patches
        stride: overlap stride
        transform: transformation to apply on patches
        mode: 'train' or 'valid'
        ntrain: fraction of images for training
        """
        super().__init__(list_of_imgs=list_of_imgs, list_of__masks=list_of__masks,#path_to_data=path_to_data,
                                transform=None,  # IMPORTANT: disable parent transform
                                mode=mode,
                                ntrain=ntrain)

        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform
        self.mode = mode

        self.image_patch_cache = []
        self.mask_patch_cache = []

        # Precompute patch index map
        self.patch_index_map = []

        if self.mode == 'train':
            for img_idx, img_path in enumerate(self.flnames_s2_img):

                ds = xr.open_dataset(img_path)
                shape = ds['Blue'].shape

                # Handle (T,H,W) or (H,W)
                if len(shape) == 3:
                    _, H, W = shape
                else:
                    H, W = shape

                ds.close()

                for i in range(0, H - patch_size + 1, stride):
                    for j in range(0, W - patch_size + 1, stride):
                        self.patch_index_map.append((img_idx, i, j))

        else:
            raise Warning('no overlap produced')
        
        # Cache
        self._cached_img_idx = None
        self._cached_img = None
        self._cached_mask = None

        
    def __len__(self):
        if self.mode == 'train':
            return len(self.patch_index_map)
        else:
            return len(self.flnames_s2_img)
        
    def __getitem__(self, idx):

        if self.mode == 'train':
            img_idx, row, col = self.patch_index_map[idx]
        else:
            img_idx = idx

        # Only reload if image changed
        if img_idx != self._cached_img_idx:

            tname_img = self.flnames_s2_img[img_idx]
            tname_mask = self.flnames_s2_mask[img_idx]

            self._cached_img = self.ds2rstr(tname_img)
            self._cached_mask = self.read_mask(tname_mask)

            self._cached_img_idx = img_idx

        timg = self._cached_img
        tmask = self._cached_mask

        if self.mode == 'train':
            timg = timg[:, :, row:row+self.patch_size, col:col+self.patch_size]
            tmask = tmask[:, row:row+self.patch_size, col:col+self.patch_size]

            if self.transform is not None:
                timg, tmask = self.transform(timg, tmask)

        return timg, tmask


def mtsk_loss(preds, labels, criterion, NClasses=1):                   
    # Multitasking loss,    segmentation / boundaries/ distance     
                                                                    
    pred_segm  = preds[:,:NClasses]                                 
    pred_bound = preds[:,NClasses:2*NClasses]                       
    pred_dists = preds[:,2*NClasses:3*NClasses]                     
                                                                    
                                                                    
                                                                    
    # Multitasking loss                                             
    label_segm  = labels[:,:NClasses]                               
    label_bound = labels[:,NClasses:2*NClasses]                     
    label_dists = labels[:,2*NClasses:3*NClasses]                   
                                                                    
                    
    #print(preds.shape, labels.shape)

    loss_segm  = criterion(pred_segm,   label_segm)                 
    loss_bound = criterion(pred_bound, label_bound)                 
    loss_dists = criterion(pred_dists, label_dists)                 
                                                                                                                                        
    return (loss_segm+loss_bound+loss_dists)/3.0 

    
def monitor_epoch(model, epoch, datagen_valid, res, criterion, NClasses=1):
    device = torch.device("cuda:0")
    metric_target = Classification(num_classes=NClasses, task='binary').to(0)
    model.eval()

    valid_pbar = tqdm(datagen_valid, desc=f"Validating Epoch {epoch}", position=1, leave=False)

    for idx, data in enumerate(valid_pbar):
        images, labels = data
        # images = images.cuda(non_blocking=True)
        # labels = labels.cuda(non_blocking=True)
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.inference_mode():
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                preds_target = model(images)
                lossi = mtsk_loss(preds_target, labels, criterion, NClasses)

        res['Epoch'].append(epoch)
        res['Iteration'].append(idx)
        res['Loss'].append(lossi.item())
        res['Mode'].append('Valid')


        pred_segm = preds_target[:, :NClasses]
        label_segm = labels[:, :NClasses]

        metric_target(pred_segm, label_segm)
     
    metric_kwargs_target = metric_target.compute()
    

    kwargs = {'epoch': epoch}
    for k, v in metric_kwargs_target.items():
        kwargs[k] = v.cpu().numpy()
    return kwargs