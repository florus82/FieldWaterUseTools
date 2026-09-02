import math
import time
from osgeo import gdal
import numpy as np
import pandas as pd
from skimage import measure
from joblib import Parallel, delayed
import os
import sys

origin = '/workspace/'
sys.path.append('/media/')

from FieldWaterUseTools.FuncBox.Misc import getFilelist, path_safe, makeTif_np_to_matching_tif, export_intermediate_products, makePyramidsForTif, stackReader
from FieldWaterUseTools.FuncBox.FieldFuncis import get_IoUs_per_Tile, subset_mask_to_prediction_extent
from FieldWaterUseTools.FuncBox.DICT_LIST import VALID_AGRO_VALUES

states = ['Brandenburg', 'Niedersachsen', 'MV', 'NRW', 'Saarland']
state_folders = ['BRB', 'LSA', 'MV', 'NRW', 'SL']


# set variables
year = 2023
model_name = 'FromScratch_IACS_dilate_True_BW_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_47'
#'IACS_dilate_True_overlap_40_on_FromScratch_IACS_dilate_True_with_overlap_47_FREEZER_2'#IACS_dilate_False_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_40_on_AI4_RGB_exclude_True_38_FREEZER_2' 
state = 'Brandenburg'
state_code = state_folders[states.index(state)]
ncores = 100
np.random.seed(42)
slicer = 10 # determines the number of tiles whole prediction will be be sliced into
border_limit = 5 # dont sample fields too close to tile borders
sample_size  = 20000
make_tifs_from_intermediate_step = False # for debugging and checks

# parameter list to check combinations for
t_exts = [i/100 for i in range(10, 30, 10)] 
t_bounds = [i/100 for i in range(10, 20, 10)]


# paths
thuenen_path = [file for file in getFilelist(f"{origin}et/Auxiliary/landcover/thuenen/", '.tif') if str(year) in file][0]
path_to_predictions = f"{origin}fields/04_Predictions/{state}/{model_name}/{year}/vrt/" # at this location, there are 
# all vrts with different chipsizes and overlaps, as well as thuenenmasked and unmasked
path_to_IACS = f"{origin}fields/01_IACS/4_Crop_mask/{state_code}/{year}/IACS_{state_code}_{year}_cropMask_cropMask_lines_touch_false_crop_touch_false_linecrop.tif" 
# all versions of the masks
# no good choices: all masks that are not linecropped 
# - cropMask_lines_touch_false_crop_touch_true --> makes fields too large
# - cropMask_lines_touch_true_crop_touch_true --> makes fields too large, no borders between close fields
# - cropMask_cropMask_lines_touch_false_crop_touch_false --> makes fields too large
# - cropMask_cropMask_lines_touch_true_crop_touch_false --> makes fields too large
# from the ones that were linecropped: 
# - cropMask_cropMask_lines_touch_false_crop_touch_false_linecrop: closest match
# - cropMask_lines_touch_false_crop_touch_true_linecrop --> 2nd best choice
# - cropMask_lines_touch_true_crop_touch_true_linecrop --> contains artefacts: fields where there are none
# - cropMask_cropMask_lines_touch_true_crop_touch_false_linecrop --> similar to cropMask_cropMask_lines_touch_false_crop_touch_false_linecrop, but less overlap

chip_overlap_mask_combos = ['masked_chips_256_20', 'unmasked_chips_256_20']#'ThuenenMask_768_20', 'unmasked_chips_768_20', 'ThuenenMask_512_20', 'unmasked_chips_512_20' 
pred_list = [f"{path_to_predictions}{combi}.vrt" for combi in chip_overlap_mask_combos]

overlords_jobs = []

ref_dim = 0

for idx, prediction in enumerate(pred_list):
    if idx == 0:
        intermediate_export = True
    else:
        intermediate_export = False
    reference_arr = subset_mask_to_prediction_extent(path_to_IACS, prediction, returnToMemory=True)

    if '/masked' in prediction:
        thuenen_arr = subset_mask_to_prediction_extent(thuenen_path, prediction, returnToMemory=True)
        th_mask = np.isin(thuenen_arr, VALID_AGRO_VALUES).astype(np.uint8)
        reference_arr[th_mask == 0] = 0 # important when we test against the thuenen masked prediction
    outFolder = path_safe(f"{origin}fields/05_GridSearch/{state}/{model_name}/{year}/{chip_overlap_mask_combos[idx]}/")

    inter_path = path_safe(f'{outFolder}intermediates3/')
    result_path = path_safe(f'{outFolder}results3/')

    ######### prepare job-list

    # create lists that will be passed on to the joblist
    # tile_list = []
    extent_true_list = []
    extent_pred_list = []
    boundary_pred_list = []
    result_dir_list = []
    row_col_start = []

    # tile predictions in prds --> total extent encompasses 90 Force Tiles (+ a few rows and cols that will be neglected as they are outside of study area)
    pred_ds = gdal.Open(prediction)

    if reference_arr.shape != ref_dim:

        rows, cols = pred_ds.RasterYSize, pred_ds.RasterXSize

        row_start = [i for i in range(0, rows, math.floor(rows/slicer))]
        row_end = [i for i in range (math.floor(rows/slicer), rows, math.floor(rows/slicer))]
        row_start = row_start[:len(row_end)] 

        col_start = [i for i in range(0, cols, math.floor(cols/slicer))]
        col_end = [i for i in range (math.floor(cols/slicer), cols, math.floor(cols/slicer))]
        col_start = col_start[:len(col_end)] 

        # label IACS reference mask
        binary_true = reference_arr > 0 # extent_true
        instances_true = measure.label(binary_true, background=0, connectivity=1)

        if make_tifs_from_intermediate_step:
            makeTif_np_to_matching_tif(instances_true, prediction, f"{inter_path}instances_true.tif", 0)
            makePyramidsForTif(f"{inter_path}instances_true.tif")
        
        # sample fields
        # build a mask to exclude fields that are in border_limit to tile borders
        power_mask = np.zeros(instances_true.shape)
        for i in range(len(row_end)):
            for j in range(len(col_end)):
                    power_mask[row_start[i]:row_start[i] + border_limit, :] = 1
                    power_mask[:, col_start[j]:col_start[j] + border_limit] = 1
                    power_mask[row_end[-1] - border_limit:power_mask.shape[0], :] = 1
                    power_mask[:, col_end[-1] - border_limit:power_mask.shape[1]] = 1

        if make_tifs_from_intermediate_step:
            makeTif_np_to_matching_tif(power_mask, prediction, f"{inter_path}powermask.tif", 0)
            makePyramidsForTif(f"{inter_path}powermask.tif")

        # get distribution of field sizes after segmentation
        unique_IDs, counts = np.unique(instances_true, return_counts=True)

        # get IDs from labelled reference that wil lbe excluded
        IDs_to_skip = np.unique(instances_true[power_mask==1])

        # exlcude fields that are too close to tile borders
        mask = ~np.isin(unique_IDs, IDs_to_skip)
        unique_IDs = unique_IDs[mask]
        counts = counts[mask]

        if make_tifs_from_intermediate_step:
            # Create filtered array with only valid IDs preserved for export
            filtered_instances = np.where(np.isin(instances_true, unique_IDs), instances_true, 0)
            makeTif_np_to_matching_tif(filtered_instances, prediction, f"{inter_path}chips_border_cut.tif", 0, gdalType=gdal.GDT_UInt32)
            makePyramidsForTif(f"{inter_path}chips_border_cut.tif")

        pixelthresh = 5
        deciles = [perc for perc in range(10,100,10)]
        mask = (unique_IDs != 0)  & (counts > pixelthresh)
        unique_IDs = unique_IDs[mask]
        counts = counts[mask]

        # get deciles and draw equally from them

        deciles_values = np.percentile(counts, deciles)
        decs = [0] + deciles_values.tolist() + [np.max(counts)]

        # exlude 0 (background) and 1 (super-small fields) from sample
        pixelthresh = 5
        deciles = [perc for perc in range(10,100,10)]
        while True:
            mask = (unique_IDs != 0)  & (counts > pixelthresh)
            unique_IDs = unique_IDs[mask]
            counts = counts[mask]

            # draw equally from decile ranges
            deciles_values = np.percentile(counts, deciles)
            decs = [0] + deciles_values.tolist() + [np.max(counts)]

            if len(decs) == len(set(decs)): # accounts for larger amount of small fields, where the deciles are the same
                df = pd.DataFrame({'decile_value': decs,
                        'excluded_pixel':pixelthresh})
                df.to_csv(f"{inter_path}decs_output.csv", index=False)
                break
            else:
                pixelthresh += 1

        bin_ids = []
        for ind in range(len(decs) -1):
            # get the unique_IDS of those fields, whose count (size) is within bin
            bin_ids.append(np.random.choice(unique_IDs[(counts > decs[ind]) & (counts <= decs[ind + 1])], int(sample_size/10), replace=False))

        instances_true = np.where(np.isin(instances_true, np.concatenate(bin_ids)),
                                instances_true,
                                0)

        if make_tifs_from_intermediate_step:
            # Create filtered array with only valid IDs preserved for export
            filtered_instances = np.where(np.isin(instances_true, unique_IDs), instances_true, 0)
            makeTif_np_to_matching_tif(filtered_instances, prediction, f"{inter_path}sampled_IDs.tif", 0, gdalType=gdal.GDT_UInt32)
            makePyramidsForTif(f"{inter_path}sampled_IDs.tif")

    ref_dim = reference_arr.shape 

    print('IDs selected - start tiling')

    # read in vrt in tiles
    for i in range(len(row_end)):
        for j in range(len(col_end)):
            
            # check for the file in result folder
            row_col_identifier = str(row_start[i]) + '_' + str(col_start[j])
            if os.path.isfile(f"{result_path}{row_col_identifier}_IoU_hyperparameter_tuning.csv"):
                continue
            ######### fill the lists with tiled data
        
            # check if tile contains a sample of reference/label data
            extent_true_label = instances_true[row_start[i]:row_end[i], col_start[j]:col_end[j]]
            if len(np.unique(extent_true_label)) == 1:
                continue
            
            extent_true_list.append(extent_true_label)
            
            #subset the prediction of fields read-in
            extent_pred_list.append(pred_ds.GetRasterBand(1).ReadAsArray(\
                col_start[j], row_start[i], col_end[j] - col_start[j], row_end[i] - row_start[i]))
                # goes into InstSegm --> image of crop probability 
            
            # load predicted boundary prob subset // goes into InstSegm --> image of boundary probability
            boundary_pred_list.append(pred_ds.GetRasterBand(2).ReadAsArray(col_start[j], row_start[i], col_end[j] - col_start[j], row_end[i] - row_start[i])) 
            
            # output folder
            result_dir_list.append(result_path)
            row_col_start.append(row_col_identifier)

            # double check
            if make_tifs_from_intermediate_step:
                export_intermediate_products(row_col_identifier, extent_pred_list[-1],\
                            pred_ds.GetGeoTransform(), pred_ds.GetProjection(), inter_path, \
                                filename='extend_pred_' + row_col_identifier + '.tif', noData=0, typ='float')

    jobs = [[row_col_start[i] ,extent_true_list[i], extent_pred_list[i], boundary_pred_list[i], result_dir_list[i], \
                pred_ds.GetGeoTransform(), pred_ds.GetProjection(), inter_path, intermediate_export, t_exts, t_bounds]  for i in range(len(result_dir_list))]

    print(f'\n{len(jobs)} tiles will be processed\n')

    del row_col_start, extent_true_list, extent_pred_list, boundary_pred_list, result_dir_list

    overlords_jobs.append(jobs)


if __name__ == '__main__':
    starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("--------------------------------------------------------")
    print("Starting process, time:" + starttime)
    print("")

    Parallel(n_jobs=ncores, temp_folder=f"{origin}tempTrash/")(delayed(get_IoUs_per_Tile)(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10]) for jobs in overlords_jobs for i in jobs)   

    print("")
    endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    print("start : " + starttime)
    print("end: " + endtime)
    print("")


    # # make vrts out 
    # t_exts = [i/100 for i in range(10,95,5)] 
    # t_bounds = [i/100 for i in range(10,95,5)]

    # ends = ['instance_pred', 'instance_true', 'intersected_at_max_and_centroids']

    # files = getFilelist(folder_path, '.tif')

    # for t_ext in t_exts:
    #     for t_bound in t_bounds:
    #         for end in ends:
    #             vrt_list = [file for file in files if f'{t_ext}_{t_bound}_{end}' in file]
    #             vrt = gdal.BuildVRT(f'{folder_path}{t_ext}_{t_bound}_{end}.vrt', vrt_list, separate = False)
    #             vrt = None
    #             convertVRTpathsTOrelative(f'{folder_path}{t_ext}_{t_bound}_{end}.vrt')
    #             vrtPyramids(f'{folder_path}{t_ext}_{t_bound}_{end}.vrt')