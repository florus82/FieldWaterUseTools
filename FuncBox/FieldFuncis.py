# some of the functions listed here were taken from https://github.com/feevos/tfcl/tree/master
# this repo is connected to the AI4boudnaries dataset https://doi.org/10.5194/essd-15-317-2023

# load functions and packages
import torch
import pickle
from tqdm import tqdm
import albumentations as A
from albumentations.core.transforms_interface import  ImageOnlyTransform
from torch.amp import autocast
import numpy as np
import rasterio
import xarray as xr
from osgeo import gdal
import os
import random
from FieldWaterUseTools.FuncBox.tfcl.utils.classification_metric import Classification
from FieldWaterUseTools.FuncBox.tfcl.models.ptavit3d import ptavit3d_dn
from FieldWaterUseTools.FuncBox.misc import getFilelist, sortListwithOtherlist, path_safe, getExtentRas, commonBoundsDim, \
    commonBoundsCoord, convertVRTpathsTOrelative, vrtPyramids


class AI4BNormal_S2(object):
    """
    class for Normalization of images, per channel, in format CHW 
    """
    def __init__(self):

        self._mean_s2 = np.array([5.4418573e+02, 7.6761194e+02, 7.1712860e+02, 2.8561428e+03 ]).astype(np.float32) 
        self._std_s2  = np.array( [3.7141626e+02, 3.8981952e+02, 4.7989127e+02 ,9.5173022e+02]).astype(np.float32) 

    def __call__(self,img):

        temp = img.astype(np.float32)
        temp2 = temp.T
        temp2 -= self._mean_s2
        temp2 /= self._std_s2

        temp = temp2.T
        return temp

  
class TrainingTransformS2(object):
    # Built on Albumentations, this provides geometric transformation only  
    def __init__(self,  prob = 1., mode='train', norm = AI4BNormal_S2() ):
        self.geom_trans = A.Compose([
                    A.RandomCrop(width=128, height=128, p=1.0),  # Always apply random crop
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
    def __init__(self, path_to_data=r'/path/to/AI4BOUNDARIES/sentinel2/',transform=TrainingTransformS2(), mode='train', ntrain=0.9):
        
        self.flnames_s2_img = getFilelist(path_to_data, '.nc', deep=True, order=True)
        self.flnames_s2_mask = getFilelist(path_to_data, '.tif', deep=True, order=True)


        assert len(self.flnames_s2_img) == len(self.flnames_s2_mask), ValueError("Some problem, the masks and images are not in the same numbers, aborting")
        
        tlen = len(self.flnames_s2_img)
        
        # Make a reproducible random split
        indices = list(range(tlen))
        random.seed(42)   # ensure reproducibility
        random.shuffle(indices)
        split_idx = int(ntrain * tlen)

        if mode == 'train':
            selected_idx = indices[:split_idx]
        elif mode == 'valid':
            selected_idx = indices[split_idx:]
        else:
            raise ValueError("Cannot undertand mode::{}, should be either train or valid, aborting...".format(mode))
        
        # Select files based on random indices
        self.flnames_s2_img = [self.flnames_s2_img[i] for i in selected_idx]
        self.flnames_s2_mask = [self.flnames_s2_mask[i] for i in selected_idx]
        self.transform=transform                                                              
    
    # Helper function to read nc to raster 
    def ds2rstr(self,tname):
        variables2use=['B2','B3','B4','B8'] # ,'NDVI']
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
    def __init__(self, path_to_data, patch_size=128, stride=64, transform=None, mode='train', ntrain=0.9):
        """
        path_to_data: root folder with images and masks
        patch_size: size of extracted patches
        stride: overlap stride
        transform: transformation to apply on patches
        mode: 'train' or 'valid'
        ntrain: fraction of images for training
        """
        super().__init__(path_to_data=path_to_data,
                                transform=None,  # IMPORTANT: disable parent transform
                                mode=mode,
                                ntrain=ntrain)

        self.patch_size = patch_size
        self.stride = stride
        self.transform = transform

        # Precompute patch index map
        self.patch_index_map = []

        for img_idx, (img_path, mask_path) in enumerate(
                zip(self.flnames_s2_img, self.flnames_s2_mask)):

            ds = xr.open_dataset(img_path)
            shape = ds['B2'].shape

            # Handle (T,H,W) or (H,W)
            if len(shape) == 3:
                _, H, W = shape
            else:
                H, W = shape

            ds.close()

            for i in range(0, H - patch_size + 1, stride):
                for j in range(0, W - patch_size + 1, stride):
                    self.patch_index_map.append((img_idx, i, j))

    def __len__(self):
        return len(self.patch_index_map)

    def __getitem__(self, idx):

        img_idx, row, col = self.patch_index_map[idx]

        tname_img = self.flnames_s2_img[img_idx]
        tname_mask = self.flnames_s2_mask[img_idx]

        # Load full image
        timg = self.ds2rstr(tname_img)           # (C, T, H, W)
        tmask = self.read_mask(tname_mask)       # (H, W, bands)


        # Slice ONE patch
        timg_patch = timg[:, :, row:row+self.patch_size, col:col+self.patch_size]

        tmask_patch = tmask[:, row:row+self.patch_size, col:col+self.patch_size]

        # Apply augmentation (ONLY here)
        if self.transform is not None:
            timg_patch, tmask_patch = self.transform(timg_patch,
                                                     tmask_patch)

        return timg_patch, tmask_patch


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


def loadVRTintoNumpyAI4(vrtPath, applyNormalizer=True):
    '''vrtPath: path in which vrts are stored
        vrts will be loaded into numpy array and normalized (for Sentinel-2 10m bands!!!!!)'''
    vrtFiles = [file for file in getFilelist(vrtPath, '.vrt') if 'Cube' not in file]
    vrtFiles = sortListwithOtherlist([int(vrt.split('_')[-1].split('.')[0]) for vrt in vrtFiles], vrtFiles)[-1]
    bands = []

    for vrt in vrtFiles:
        ds = gdal.Open(vrt)
        bands.append(ds.GetRasterBand(1).ReadAsArray())
    cube = np.dstack(bands)
   
    data_cube = np.transpose(cube, (2, 0, 1))
    reshaped_cube = data_cube.reshape(4, 6, ds.RasterYSize, ds.RasterXSize)
    normalizer = AI4BNormal_S2()
    if applyNormalizer:
        return normalizer(reshaped_cube)
    else:
        return reshaped_cube
    

def predict_on_GPU(path_to_model, list_of_row_col_indices, npdstack, batch_size=1, temp_path = False):
    '''
    path_to_model: path to .pth file
    list_of_row_col_indices: a list in the order row_start, row_end, col_start, col_end (output of get_row_col_indices). This will be used to read in small chips from npdstack
    npdstack: normalized sentinel-2 npdstack (output from loadVRTintoNUmpyAI4)
    '''

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
    model = ptavit3d_dn(**model_config).to(device)
    model.load_state_dict(torch.load(path_to_model, map_location=device))
    model.eval()
   
    row_start, row_end, col_start, col_end = list_of_row_col_indices

    preds = []
    patches = []
    # cut np array into patches
    for i in range(len(row_end)):
        for j in range(len(col_end)):
            patches.append(npdstack[np.newaxis, :, :, row_start[i]:row_end[i], col_start[j]:col_end[j]])

    # inference
    with torch.inference_mode():
        with torch.cuda.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            for k in range(0, len(patches), batch_size):
                batch_np = np.concatenate(patches[k:k+batch_size], axis=0)
                batch_tensor = torch.from_numpy(batch_np).float().to(device, non_blocking=True)
                batch_pred = model(batch_tensor)
                preds.append(batch_pred.cpu())  

    preds = torch.cat(preds, dim=0).numpy()

    # image = torch.tensor()
    # image = image.to(torch.float)
    # image = image.to(device)  # Move image to the correct device

    # with torch.no_grad():
    #     pred = model(image)
    #     preds.append(pred.detach().cpu().numpy())
                
    # torch.cuda.empty_cache()
    # del model
    # del modeli
    # del device
    # del image

    if temp_path:
        with open(path_safe(f'{temp_path}preds.pkl'), 'wb') as f:
            pickle.dump(preds, f)
    # Load again
    # with open(f'{temp_path}preds.pkl', 'rb') as f:
    #     preds = pickle.load(f)

    return preds


def subset_mask_to_prediction_extent(path_reference_mask, path_to_prediction_vrt):
    '''
    path_reference_mask: path to the reference mask
    path_to_prediction_vrt: path to a vrt of the predicted image chips
    '''

    # check if mask has different extent from prediction
    # if so, make it the same extent for further processing (classification)
    # --> mask can never be smaller than prediciton, therefore no need to check

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


            out_ds = gdal.GetDriverByName('GTiff').Create(path_reference_mask.split('.')[0] + '_prediction_extent.tif', 
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


def export_GPU_predictions(list_of_predictions, path_to_mask, vrt_path, list_of_row_col_indices, out_path, chipsize, overlap):
    '''
    list_of_predictions: a list of predicted chips at same dimensions (output from predict_on_GPU
    path_to_mask: a path to mask that has the same dimensions as the vrt on which predictions have been undertaken; can be also a list of masks
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

        arr = list_of_predictions[i][0].transpose(1, 2, 0)
        for band in range(3):
            out_ds.GetRasterBand(band + 1).WriteArray(arr[int(overlap/2): -int(overlap/2), int(overlap/2): -int(overlap/2), band])
        del out_ds

    print('umnasked chips exported')

    # check if mask is a list or single mask
    if isinstance(path_to_mask, list):
        print('list of masks provided - start exporting')
        for maski in path_to_mask:
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

                arr = list_of_predictions[i][0].transpose(1, 2, 0)

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