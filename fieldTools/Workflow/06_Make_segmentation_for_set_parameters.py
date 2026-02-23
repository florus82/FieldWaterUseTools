import sys

origin = '/workspace/'
sys.path.append('/media/')

import math
from FieldWaterUseTools.FuncBox.Misc import path_safe, dirfinder, get_row_col_indices
from FieldWaterUseTools.FuncBox.FieldFuncis import *


#########################  parameter settings
# load the prediction and labels
year = 2023
model_name = 'AI4_RGB_exclude_True_38'
ncores = 56
all_touch_lines = 'true'
all_touch_field = 'false'
state = 'Brandenburg'

t_exts = [0.1] 
t_bounds = [0.5]


# load precitions
unmasked = [file for file in getFilelist(f"{origin}fields/output/predictions/FORCE/BRANDENBURG/{model_name}/{year}/vrt/", '.vrt')\
             if 'unmasked' in file][0]

masked = [file for file in getFilelist(f"{origin}fields/output/predictions/FORCE/BRANDENBURG/{model_name}/{year}/vrt/", '.vrt')\
             if f'lines_touch_{all_touch_lines}_crop_touch_{all_touch_field}_linecrop' in file][0]

ref = [file for file in getFilelist(f"{origin}IACS/4_Crop_mask/{year}/", '.tif') if 'prediction_extent' in file and
       f'lines_touch_{all_touch_lines}_crop_touch_{all_touch_field}_linecrop' in file][0]

t_exts = [0.1] * len(ref)
t_bounds = [0.65] * len(ref)

for prediction, reference, t_ext, t_bound in zip([unmasked, masked], [ref, ref], t_exts, t_bounds):
#for prediction, reference, t_ext, t_bound in zip([unmasked]*len(ref), ref, t_exts, t_bounds):

    if 'unmasked' in prediction:
        result_dir = path_safe(
            f"{origin}fields/segmented/{state}/{model_name}/{year}/unmasked_text{t_ext}_tbound{t_bound}/"#{reference.split('/')[-1].split('GSA-DE_BRB-2023_cropMask_')[-1].split('_prediction')[0]}/"
            )
    else:
        result_dir = path_safe(
            f"{origin}segmented/{state}/{model_name}/{year}/masked_lines_touch_{all_touch_lines}_crop_touch_{all_touch_field}_linecrop_text{t_ext}_tbound{t_bound}/"
            )
        
    folder_path = path_safe(f'{result_dir}/intermediates/')
    results_folder_path = path_safe(f'{result_dir}/results/')
    
    # set the number by which rows and cols will be divided --> determines the number of tiles // also set border limit (dont sample fields too close to tile borders) and sample size
    slicer = 10
    border_limit = 5
    sample_size  = 10000
    # set the number of cores for parallel processing and set seed
    np.random.seed(42)
    make_tifs_from_intermediate_step = True

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
    rows, cols = pred_ds.RasterYSize, pred_ds.RasterXSize

    row_start = [i for i in range(0, rows, math.floor(rows/slicer))]
    row_end = [i for i in range (math.floor(rows/slicer), rows, math.floor(rows/slicer))]
    row_start = row_start[:len(row_end)] 

    col_start = [i for i in range(0, cols, math.floor(cols/slicer))]
    col_end = [i for i in range (math.floor(cols/slicer), cols, math.floor(cols/slicer))]
    col_start = col_start[:len(col_end)] 


    # load IACS reference mask and label it 
    ref_ds = gdal.Open(reference)
    extent_true = ref_ds.GetRasterBand(1).ReadAsArray()
    binary_true = extent_true > 0
    instances_true = measure.label(binary_true, background=0, connectivity=1)

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
        makeTif_np_to_matching_tif(power_mask, reference, folder_path + 'powermask.tif', 0)
        makePyramidsForTif(folder_path + 'powermask.tif')

    # get IDs from labelled reference
    IDs_to_skip = np.unique(instances_true[power_mask==1])

    # get distribution of field sizes after segmentation
    unique_IDs, counts = np.unique(instances_true, return_counts=True)

    # exlcude fields that are too close to tile borders
    mask = ~np.isin(unique_IDs, IDs_to_skip)
    unique_IDs = unique_IDs[mask]
    counts = counts[mask]

    if make_tifs_from_intermediate_step:
        # Create filtered array with only valid IDs preserved for export
        filtered_instances = np.where(np.isin(instances_true, unique_IDs), instances_true, 0)
        makeTif_np_to_matching_tif(filtered_instances, reference, folder_path + 'chips_border_cut.tif', 0, gdalType=gdal.GDT_UInt32)
        makePyramidsForTif(folder_path + 'chips_border_cut.tif')

    # exlude 0 (background) and 1 (super-small fields) from sample
    pixelthresh = 0
    while True:
        mask = (unique_IDs != 0)  & (counts > pixelthresh)
        unique_IDs = unique_IDs[mask]
        counts = counts[mask]

        # get deciles and draw equally from them
        deciles = [perc for perc in range(10,100,10)]
        deciles_values = np.percentile(counts, deciles)
        decs = [0] + deciles_values.tolist() + [np.max(counts)]

        if len(decs) == len(set(decs)):
            df = pd.DataFrame({'decile_value': decs,
                   'excluded_pixel':pixelthresh})
            df.to_csv(f'{folder_path}decs_output.csv', index=False)
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
        makeTif_np_to_matching_tif(filtered_instances, reference, folder_path + 'valid_IDs.tif', 0, gdalType=gdal.GDT_UInt32)
        makePyramidsForTif(folder_path + 'valid_IDs.tif')

    print('IDs selected - start tiling')

    # read in vrt in tiles
    for i in range(len(row_end)):
        for j in range(len(col_end)):
            
            ######### fill the lists with tiled data

            #subset the prediction of fields read-in
            extent_pred = pred_ds.GetRasterBand(1).ReadAsArray(col_start[j], row_start[i], col_end[j] - col_start[j], row_end[i] - row_start[i]) # goes into InstSegm --> image of crop probability 
            # # mask extend_pred with reference
            # extent_pred_masked = extent_pred * extent_true[row_start[i]:row_end[i], col_start[j]:col_end[j]]

            # # check if prediction subset of fields actually contains data
            # if len(np.unique(extent_pred_masked)) == 1:
            #     continue

            # check if tile contains a sample of reference/label data
            extent_true_label = instances_true[row_start[i]:row_end[i], col_start[j]:col_end[j]]
            if len(np.unique(extent_true_label)) == 1:
                continue
            
            extent_true_list.append(extent_true_label)
            # extent_pred_list.append(extent_pred_masked)
            extent_pred_list.append(extent_pred)
            
            # make identifier for tile for csv
            # tile_list.append(f'{str(i)}_{str(j)}')
            # load predicted boundary prob subset // goes into InstSegm --> image of boundary probability
            boundary_pred_list.append(pred_ds.GetRasterBand(2).ReadAsArray(col_start[j], row_start[i], col_end[j] - col_start[j], row_end[i] - row_start[i])) 
            # output folder
            result_dir_list.append(results_folder_path)
            row_col_start.append(str(row_start[i]) + '_' + str(col_start[j]))

            # double check
            if make_tifs_from_intermediate_step:
                # export_intermediate_products(str(row_start[i]) + '_' + str(col_start[j]), extent_pred_masked, pred_ds.GetGeoTransform(), pred_ds.GetProjection(),\
                #                     folder_path, filename='extend_pred_masked_false_' + str(row_start[i]) + '_' + str(col_start[j]) + '.tif', noData=0, typ='float')
                
                export_intermediate_products(str(row_start[i]) + '_' + str(col_start[j]), extent_pred, pred_ds.GetGeoTransform(), pred_ds.GetProjection(),\
                                    folder_path, filename='extend_pred_false_' + str(row_start[i]) + '_' + str(col_start[j]) + '.tif', noData=0, typ='float')

    jobs = [[row_col_start[i] ,extent_true_list[i], extent_pred_list[i], boundary_pred_list[i], result_dir_list[i],\
             pred_ds.GetGeoTransform(), pred_ds.GetProjection(), folder_path, True, t_ext, t_bound]  for i in range(len(result_dir_list))]
    # tile_list[i], 
    print(f'\n{len(jobs)} tiles will be processed\n')

    del row_col_start, extent_true_list, extent_pred_list, boundary_pred_list, result_dir_list, border_limit

    # overlords_jobs.append(jobs)


    if __name__ == '__main__':
        starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
        print("--------------------------------------------------------")
        print("Starting process, time:" + starttime)
        print("")

        Parallel(n_jobs=ncores)(delayed(get_IoUs_per_Tile)(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8], i[9], i[10]) for i in jobs)   # for jobs in overlords_jobs 

        print("")
        endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
        print("--------------------------------------------------------")
        print("--------------------------------------------------------")
        print("start : " + starttime)
        print("end: " + endtime)
        print("")
