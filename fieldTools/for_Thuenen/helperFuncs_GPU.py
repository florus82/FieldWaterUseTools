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



############################# for prediction

def predict_on_GPU_without_preload(path_to_model, list_of_row_col_indices, list_of_vrts, temp_path = False):
    '''
    path_to_model: path to .pth file
    list_of_row_col_indices: a list in the order row_start, row_end, col_start, col_end (output of get_row_col_indices). This will be used to read in small chips from npdstack
    list_of_vrtFiles for input stack: has to be read-in and normalized
    '''

    normalizer = AI4BNormal_S2()

    row_start = list_of_row_col_indices[0]
    row_end   = list_of_row_col_indices[1]
    col_start = list_of_row_col_indices[2]
    col_end   = list_of_row_col_indices[3]

    # define the model (.pth) and assess loss curves
    #model_name = dataFolder + 'output/models/model_state_All_but_LU_transformed_42.pth'
    model_name_short = path_to_model.split('/')[-1].split('.')[0]
 
    NClasses = 1
    nf = 96
    verbose = True
    model_config = {'in_channels': 4,
                    'spatial_size_init': (128, 128),
                    'depths': [2, 2, 5, 2],
                    'nfilters_init': nf,
                    'nheads_start': nf // 4,
                    'NClasses': NClasses,
                    'verbose': verbose,
                    'segm_act': 'sigmoid'}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        modeli = ptavit3d_dn(**model_config).to(device)
        modeli.load_state_dict(torch.load(path_to_model))
        model = modeli.to(device) # Set model to gpu
        model.eval()
        
    preds = []

    for i in range(len(row_end)):
        for j in range(len(col_end)):
            bands = []
            for vrt in list_of_vrts:
                ds = gdal.Open(vrt, gdal.GA_ReadOnly)
                bands.append(
                    ds.GetRasterBand(1).ReadAsArray(col_start[j], row_start[i], col_end[j] - col_start[j], row_end[i] - row_start[i])
                )
            cube = np.dstack(bands)  # (y, x, bands)

            data_cube = np.transpose(cube, (2, 0, 1))
            reshaped_cube = data_cube.reshape(4, 6, cube.shape[0], cube.shape[1])
            
            norm_cube = normalizer(reshaped_cube)

            image = torch.tensor(norm_cube) # npdstack[np.newaxis, :, :, row_start[i]:row_end[i], col_start[j]:col_end[j]])
            image = image.to(torch.float)
            image = image.unsqueeze(0).to(device)  # Move image to the correct device
        
            with torch.no_grad():
                pred = model(image)
                preds.append(pred.detach().cpu().numpy())

                print(f"{i} from {len(row_end)} and {j} from {len(col_end)}")
                
    torch.cuda.empty_cache()
    del model
    del modeli
    del device
    del image

    if temp_path:
        with open(path_safe(f'{temp_path}preds.pkl'), 'wb') as f:
            pickle.dump(preds, f)

    # Load again
    # with open(f'{temp_path}preds.pkl', 'rb') as f:
    #     preds = pickle.load(f)

    return preds


def export_GPU_predictions(list_of_predictions, path_to_mask, vrt_path, list_of_row_col_indices, out_path, chipsize, overlap):
    '''
    list_of_predictions: a list of predicted chips at same dimensions (output from predict_on_GPU
    path_to_mask: a path to mask that has the same dimensions as the vrt on which predictions have been
                undertaken; can be also a list of masks; can also be a cached object
    vrt_path: path to a folder that contains the vrt files, the predictions (and mask) is based on. Will be used for GeoTransform and Projection
    list_of_row_col_indices: a list in the order row_start, row_end, col_start, col_end (output of get_row_col_indices). 
                                Will be used to read in mask chips and manipulate Geotransform
    out_path: path to where the predicted images should be stored to
    '''

    row_start = list_of_row_col_indices[0]
    row_end   = list_of_row_col_indices[1]
    col_start = list_of_row_col_indices[2]
    col_end   = list_of_row_col_indices[3]

    if not out_path.endswith('/'):
        out_path = out_path + '/'

    gtiff_driver = gdal.GetDriverByName('GTiff')
    vrt_ds = gdal.Open(getFilelist(vrt_path, '.vrt')[0])
    geoTF = vrt_ds.GetGeoTransform()
    filenames = [f'X_{col_start[j]}_Y_{row_start[i]}.tif' for i in range(len(row_start)) for j in range(len(col_start))]

    # export unmasked chips
    for i, file in enumerate(filenames):
        out_ds = gtiff_driver.Create(path_safe(f'{out_path}unmasked_chips/chips_unmasked{str(chipsize)}_{overlap}_{file}'), int(chipsize - overlap), int(chipsize - overlap), 3, gdal.GDT_Float32)
        # change the Geotransform for each chip
        geotf = list(geoTF)
        # get column and rows from filenames
        geotf[0] = geotf[0] + geotf[1] * (int(file.split('X_')[-1].split('_')[0]) + overlap/2)
        geotf[3] = geotf[3] + geotf[5] * (int(file.split('Y_')[-1].split('.')[0]) + overlap/2)
        #print(f'X:{geoTF[0]}  Y:{geoTF[3]}  AT {file}')
        out_ds.SetGeoTransform(tuple(geotf))
        out_ds.SetProjection(vrt_ds.GetProjection())

        arr = list_of_predictions[i][0].transpose(1, 2, 0) # [i][0]
        for band in range(3):
            out_ds.GetRasterBand(band + 1).WriteArray(arr[int(overlap/2): -int(overlap/2), int(overlap/2): -int(overlap/2), band])
        del out_ds

    print('umnasked chips exported')

    # check if mask is a list or single mask
    if isinstance(path_to_mask, list):
        pass
    elif path_to_mask == 'no mask':
        return
    else:
        path_to_mask = [path_to_mask]

    for maski in path_to_mask:
        if isinstance(maski, np.ndarray):
            mask_name = 'ThuenenMask'
            mask = maski
        else:
            mask_name = maski.split('cropMask_')[-1].split('.')[0]            
            # load mask
            ds = gdal.Open(maski)
            mask = ds.GetRasterBand(1).ReadAsArray()

        for i, file in enumerate(filenames):
            out_ds = gtiff_driver.Create(path_safe(f'{out_path}{mask_name}/{mask_name}_{str(chipsize)}_{overlap}_{file}'), int(chipsize - overlap), int(chipsize - overlap), 3, gdal.GDT_Float32)
            # change the Geotransform for each chip
            geotf = list(geoTF)
            # get column and rows from filenames
            geotf[0] = geotf[0] + geotf[1] * (int(file.split('X_')[-1].split('_')[0]) + overlap/2)
            geotf[3] = geotf[3] + geotf[5] * (int(file.split('Y_')[-1].split('.')[0]) + overlap/2)
            #print(f'X:{geoTF[0]}  Y:{geoTF[3]}  AT {file}')
            out_ds.SetGeoTransform(tuple(geotf))
            out_ds.SetProjection(vrt_ds.GetProjection())

            arr = list_of_predictions[i][0].transpose(1, 2, 0) # [i][0]

            maskSub = mask[int(int(file.split('Y_')[-1].split('.')[0]) + overlap/2):chipsize + int(int(file.split('Y_')[-1].split('.')[0]) - overlap/2), 
                        int(int(file.split('X_')[-1].split('_')[0]) + overlap/2):chipsize + int(int(file.split('X_')[-1].split('_')[0]) - overlap/2)]
            for band in range(3):                
                out_ds.GetRasterBand(band + 1).WriteArray(arr[int(overlap/2): -int(overlap/2), int(overlap/2): -int(overlap/2), band] * maskSub)
            del out_ds

def predicted_chips_to_vrt(path_to_chips, chipname, chipsize, overlap, path_to_folder_out, pyramids=False):
    '''
    path_to_chips: path to chips exported via export_GPU_predictions
    chipsize + overlap: the size of these chips (in order to select the chips if chips from different predictions are in the same folder)
    path_to_folder_out: path to FOLDER where vrt will be stored
    '''
    if not path_to_folder_out.endswith('/'):
        path_to_folder_out = path_to_folder_out + '/'
    if not path_to_chips.endswith('/'):
        path_to_chips = path_to_chips + '/'  
    path_to_chips = f'{path_to_chips}{chipname}/'
    os.makedirs(path_to_folder_out, exist_ok=True)

    chip_id = f'{chipsize}_{overlap}'
    chips = getFilelist(path_to_chips, '.tif')
    chips = [chip for chip in chips if chip_id in chip]

    # for c in chips:print(c)
    # create stacked vrts of chips
    vrt_name = f'{path_to_folder_out}{chipname}_{chipsize}_{overlap}.vrt'
    vrt = gdal.BuildVRT(vrt_name, chips, separate = False)
    vrt = None
    convertVRTpathsTOrelative(vrt_name)

    if pyramids:
        vrtPyramids(vrt_name)


def subset_mask_to_prediction_extent(path_reference_mask, path_to_prediction_vrt, area='no_area_specified', returnToMemory=False):
    '''
    path_reference_mask: path to the reference mask
    path_to_prediction_vrt: path to a vrt of the predicted image chips
    '''

    # check if mask has different extent from prediction
    # if so, make it the same extent for further processing (classification)
    # --> mask can never be smaller than prediciton, therefore no need to check # not true for extent of germany

    ext_mask = getExtentRas(path_reference_mask)
    ext_pred = getExtentRas(path_to_prediction_vrt)

    if ext_mask == ext_pred:
        print('Mask already has same extent as prediction - no further subsetting needed :)')
    else:
        common_bounds = commonBoundsDim([ext_mask, ext_pred])
        common_coords = commonBoundsCoord(common_bounds)
        if common_bounds == ext_pred:
            ds = gdal.Open(path_reference_mask)
            in_gt = ds.GetGeoTransform()
            inv_gt = gdal.InvGeoTransform(in_gt)
            # transform coordinates into offsets (in cells) and make them integer
            off_UpperLeft = gdal.ApplyGeoTransform(inv_gt, common_coords[0]['UpperLeftXY'][0], common_coords[0]['UpperLeftXY'][1])  # new UL * rastersize^-1  + original ul/rastersize(opposite sign
            off_LowerRight = gdal.ApplyGeoTransform(inv_gt, common_coords[0]['LowerRightXY'][0], common_coords[0]['LowerRightXY'][1])
            off_ULx, off_ULy = map(round, off_UpperLeft) 
            off_LRx, off_LRy = map(round, off_LowerRight)

            band = ds.GetRasterBand(1)
            data = band.ReadAsArray(off_ULx, off_ULy, off_LRx - off_ULx, off_LRy - off_ULy)

            if not returnToMemory:
                out_ds = gdal.GetDriverByName('GTiff').Create(f"{path_reference_mask.split('.')[0]}_prediction_extent_{area}.tif", 
                                                            off_LRx - off_ULx, 
                                                            off_LRy - off_ULy, 1, ds.GetRasterBand(1).DataType)
                out_gt = list(in_gt)
                out_gt[0], out_gt[3] = gdal.ApplyGeoTransform(in_gt, off_ULx, off_ULy)
                out_ds.SetGeoTransform(out_gt)
                out_ds.SetProjection(ds.GetProjection())

                out_ds.GetRasterBand(1).WriteArray(data)
                if band.GetNoDataValue():
                    out_ds.GetRasterBand(1).SetNoDataValue(band.GetNoDataValue())
                del out_ds
            else:
                return data
