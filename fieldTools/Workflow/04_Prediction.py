import sys
origin = '/workspace/'
sys.path.append('/media/')

from tqdm import tqdm
from FieldWaterUseTools.FuncBox.Misc import path_safe, dirfinder, get_row_col_indices, checkPath, getFilelist,\
     stackReader, warp_np_to_reference
from FieldWaterUseTools.FuncBox.FieldFuncis import *
from FieldWaterUseTools.FuncBox.DICT_LIST import VALID_AGRO_VALUES


####################################################### Prepare
states = ['Brandenburg', 'Niedersachsen', 'MV', 'NRW', 'Saarland']
state_folders = ['BRB', 'LSA', 'MV', 'NRW', 'SL']
model_names = ['model_state_IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_40_on_FromScratch_IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_47_FREEZER_2']
# ['model_state_FromScratch_IACS_dilate_False_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_22',
#                'model_state_FromScratch_IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_47',
#                'model_state_IACS_dilate_False_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_40_on_AI4_RGB_exclude_True_38_FREEZER_2',
#                'model_state_IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_40_on_AI4_RGB_exclude_True_38_FREEZER_2'] 
#'model_state_AI4_RGB_exclude_True_38'


for model_name in model_names:

     year = 2023
     state = 'Brandenburg'
     # set tiling scheme and chip size on which prediction will be undertaken
     chipsize = 128*2 # 5 or 6 is the maximum with GPU in basement
     overlap  = 20

     # set paths
     state_code = state_folders[states.index(state)]
     predict_master_folder = path_safe(f"{origin}fields/04_Predictions/{state}/{model_name.split('_state_')[-1]}/{year}/")
     vrt_path = f"{origin}fields/Misc/S2vrt/{state_code}/{year}/"
     vrt_Folder = f"{vrt_path}{dirfinder(vrt_path)[0]}"
     cropMask_Folder = f"{origin}fields/01_IACS/4_Crop_mask/{state_code}/{year}/"
     thuenen_path = [file for file in getFilelist(f"{origin}et/Auxiliary/landcover/thuenen/", '.tif') if str(year) in file][0]

     # load vrts into npdstack
     print('Load vrt into numpy for prediction')
     dat = loadVRTintoNumpyAI4(vrt_Folder)
     # print(f'loaded np array has the shape:{dat.shape}')

     row_col_ind = get_row_col_indices(chipsize, overlap, dat.shape[2:][0], dat.shape[2:][1])

     # ####################################################### Predict
     print('start prediction')
     predicted_chips_list = predict_on_GPU(f'{origin}fields/03_Output/models/{model_name}.pth', row_col_ind, dat, 
                                        temp_path=f'{predict_master_folder}temp/', batch_size=1)# model_state_All_but_LU_transformed_42

     ####################################################### Postprocess
     print('start exporting')

     # load Thuenen mask and create binary mask
     th_arr = stackReader(thuenen_path)
     th_mask = np.isin(th_arr, VALID_AGRO_VALUES).astype(np.uint8)

     print(vrt_Folder)
     th_masked_warped = warp_np_to_reference(th_mask,thuenen_path, getFilelist(vrt_Folder, '.vrt')[0], resamp=gdal.GRA_NearestNeighbour)
     # export the predicted chips (masked and not masked)

     # with open(f'{predict_master_folder}temp/preds.pkl', 'rb') as f:
     #     predicted_chips_list = pickle.load(f)

     export_GPU_predictions(predicted_chips_list, 
                         #[crop_mask_file for crop_mask_file in \
                         # getFilelist(cropMask_Folder, '.tif') if 'prediction_extent' not in crop_mask_file], #f'{prefix}fields/IACS/4_Crop_mask/{year}/GSA-DE_BRB-{year}_cropMask_lines_touch_false_lines_touch_false_linecrop.tif', 
                         th_masked_warped,
                         vrt_Folder,
                         row_col_ind, 
                         path_safe(f'{predict_master_folder}chips_folder/'),
                         chipsize, overlap)


     # make vrt of predicted image chips
     for chip in dirfinder(f'{predict_master_folder}chips_folder/'):
          predicted_chips_to_vrt(f'{predict_master_folder}chips_folder/', chip,  chipsize, overlap,
                              path_safe(f'{predict_master_folder}vrt/'), pyramids=True)


     # # adapt for case that extent_prection is already in there !!!

     # make a subset of the reference mask (extent FORCE output) to the extent of the prediction
     # for crop_mask_file in getFilelist(cropMask_Folder, '.tif'): 
     #     subset_mask_to_prediction_extent(crop_mask_file,
     #     getFilelist(f'{predict_master_folder}vrt/', '.vrt')[0])
     # subset_mask_to_prediction_extent(thuenen_path, getFilelist(f'{predict_master_folder}vrt/', '.vrt')[0], state)