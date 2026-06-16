import sys

origin = '/workspace/'
sys.path.append('/media/')

from joblib import Parallel, delayed
import math
from FieldWaterUseTools.FuncBox.Misc import path_safe, dirfinder, get_row_col_indices
from FieldWaterUseTools.FuncBox.FieldFuncis import *

#########################  parameter settings
ncores = 120
overlord_jobs = []
# load the prediction and labels
year = 2023
state = 'GERMANY'

model_name = 'FromScratch_IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_47'
mod_id = 'FromScratch_dilate_T'
folder_name = 'MaskedThuenen'
file_name = 'Masked_THUENEN_CTM_2023'
path_to_prediction = f"{origin}fields/04_Predictions/{state}/{model_name}/{year}/vrt/{file_name}.tif"


t_exts_list = [[.7]]

t_bounds_list = [[.2]]

trash_path = f"{origin}tempTrash/"

for t_exts, t_bounds in zip(t_exts_list, t_bounds_list):
    
    for idx2 in range(len(t_exts)):
        t_ext = t_exts[idx2]
        t_bound = t_bounds[idx2]
        result_dir = path_safe(f"{origin}fields/06_Segmentation/{state}/{mod_id}/{year}/{folder_name}/Tiles/")
        seg_export_name = f"{file_name}_segmented_ext_{str(t_ext).replace('.', '')}_bound_{str(t_bound).replace('.', '')}"

        # set the number by which rows and cols will be divided --> determines the number of tiles
        slicer = 75
        # set the number of cores for parallel processing and set seed
        np.random.seed(42)

        ######### prepare job-list

        # create lists that will be passed on to the joblist
        extent_pred_list = []
        boundary_pred_list = []
        result_dir_list = []
        row_col_start = []
        seg_name_list = []


        # tile predictions in prds --> total extent encompasses 90 Force Tiles (+ a few rows and cols that will be neglected as they are outside of study area)
        pred_ds = gdal.Open(path_to_prediction)
        rows, cols = pred_ds.RasterYSize, pred_ds.RasterXSize

        row_start = [i for i in range(0, rows, math.floor(rows/slicer))]
        row_end = [i for i in range (math.floor(rows/slicer), rows, math.floor(rows/slicer))]
        row_start = row_start[:len(row_end)] 

        col_start = [i for i in range(0, cols, math.floor(cols/slicer))]
        col_end = [i for i in range (math.floor(cols/slicer), cols, math.floor(cols/slicer))]
        col_start = col_start[:len(col_end)] 


        print('Start tiling')

        # read in vrt in tiles
        for i in range(len(row_end)):
            for j in range(len(col_end)):
                
                ######### fill the lists with tiled data

                #subset the prediction of fields read-in
                extent_pred_list.append(pred_ds.GetRasterBand(1).ReadAsArray(col_start[j], row_start[i], col_end[j] - col_start[j], row_end[i] - row_start[i])) # goes into InstSegm --> image of crop probability 
                # load predicted boundary prob subset // goes into InstSegm --> image of boundary probability
                boundary_pred_list.append(pred_ds.GetRasterBand(2).ReadAsArray(col_start[j], row_start[i], col_end[j] - col_start[j], row_end[i] - row_start[i])) 
                # output folder
                result_dir_list.append(result_dir)
                seg_name_list.append(seg_export_name)
                row_col_start.append(str(row_start[i]) + '_' + str(col_start[j]))

            

        jobs = [[row_col_start[i], extent_pred_list[i], boundary_pred_list[i], result_dir_list[i], seg_name_list[i],
                    pred_ds.GetGeoTransform(), pred_ds.GetProjection(), t_ext, t_bound]  for i in range(len(result_dir_list))]


        print(f'\n{len(jobs)} tiles will be processed\n')


        del row_col_start, extent_pred_list, boundary_pred_list, result_dir_list, seg_name_list


        overlord_jobs.append(jobs)


if __name__ == '__main__':
    starttime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("--------------------------------------------------------")
    print("Starting process, time:" + starttime)
    print("")

    Parallel(n_jobs=ncores, temp_folder=trash_path)(delayed(apply_seg_parameters)(i[0], i[1], i[2], i[3], i[4], i[5], i[6], i[7], i[8]) for jobs in overlord_jobs for i in jobs)

    print("")
    endtime = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print("--------------------------------------------------------")
    print("--------------------------------------------------------")
    print("start : " + starttime)
    print("end: " + endtime)
    print("")


# make vrts
for t_exts, t_bounds in zip(t_exts_list, t_bounds_list):

    for idx2 in range(len(t_exts)):
        t_ext = t_exts[idx2]
        t_bound = t_bounds[idx2]
        suffix = f"ext_{str(t_ext).replace('.', '')}_bound_{str(t_bound).replace('.', '')}"
        tiles = [file for file in getFilelist(f"{origin}fields/06_Segmentation/{state}/{mod_id}/{year}/{folder_name}/Tiles/", '.tif') if suffix in file]
        
        if tiles:
            vrt_path = path_safe(f"{origin}fields/06_Segmentation/{state}/{mod_id}/{year}/{folder_name}/vrt/")
            vrt_out = f"{vrt_path}{folder_name}_{suffix}.vrt"
            vrt = gdal.BuildVRT(vrt_out, tiles, separate = False)
            vrt = None

            convertVRTpathsTOrelative(vrt_out)