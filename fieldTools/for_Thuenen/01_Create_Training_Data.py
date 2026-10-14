import sys

origin = '/workspace/'
sys.path.append('/media/')

from tqdm import tqdm
from torch.amp import autocast, GradScaler
import pandas as pd
from datetime import datetime

import rioxarray       
from FieldWaterUseTools.FuncBox.FieldFuncis import *
from FieldWaterUseTools.FuncBox.ForceFuncis import force_order_Colors_for_VRT, force_to_vrt, reduce_forceTSA_output_to_validmonths, \
    force_to_vrt
from FieldWaterUseTools.FuncBox.Polygons_to_Labels import *
from FieldWaterUseTools.FuncBox.Misc import get_row_col_indices, stackReader, stack_tifs, dirfinder
from FieldWaterUseTools.FuncBox.DICT_LIST import EXCLUDE_LIST


################################################ create the images as nc and labels as tif files

## set paths

# specify the path to the folder where all IACS datasets are stored (it is assumed that all datasets are in one folder)
IACS_path = f'{origin}fields/01_IACS/1_Polygons/'
# specify the output folder for the training data
out_folder = path_safe(f'{origin}fields/02_Chips_generated_from_IACS/')
# specify a tempFolder: here will be stored rasterized IACS data before cutting it into chips
temp_folder = path_safe(f'{origin}fields/temp_trash/')
# specify the folder where FORCE output is stored. The FORCE tiles are expected to be located inside specifying the year of observation that are inside
# folders with state names (see variable "state_folders" right below)
FORCE_folder = f'{origin}force/output/'
# specify the folder where FORCE vrts will be stored
aux_vrt_path = f'{origin}fields/Auxiliary/vrt/'


## set states and years

# specify the states that should be included in the training dataset. Please note, that these names are used for folders to be created and printing purposes
states = ['Brandenburg', 'Niedersachsen', 'MV', 'NRW', 'Saarland']
# specify the foldernames FORCE tiles are stored in (see path "FORCE_folder" above). Order must be the same as variable "states"
state_folders = ['BRB', 'LSA', 'MV', 'NRW', 'SL']
# specify the file.extensions of IACS file, matching the order of "states"/"state_folders"
state_types = ['.geoparquet','.geoparquet','.shp','.geoparquet','.shp']
# if IACS files will be subsetted for certain attributes, specify the column of the respective IACS's data set's attribute table, where this attribute can be located
state_exclude_columns = ['EC_hcat_n', 'EC_hcat_n', 'NU_BEZ', 'EC_hcat_n', 'BEZ']
# specify the year in which FORCE tiles are within "state_folders"
years = [2022, 2024, 2023, 2020, 2021]

## set chips, sample size, seed, min thresh
chip_size = 256
Total_number_of_samples = 4000
seed = 42
min_pix_thresh = 10 # minimum number of pixel (either field or border) that should be present in a sample chip will be larger than this value

# FIXED variables: the following variables are used to identify single FORCE files, as well as variable info for nc files
colorL = ['BLU', 'GRN', 'RED', 'BNR']#['BLU', 'GRN', 'RED', 'NIR']
gtiff_driver = gdal.GetDriverByName('GTiff')
random.seed(seed)


################################ Rasterize IACS files

for state, state_folder, state_type, state_exclude_column, year in\
    zip(states, state_folders, state_types, state_exclude_columns, years): # state_burner, state_burners,

    print(state)
    path = [file for file in getFilelist(IACS_path + state_folder, state_type) if str(year) in file][0]
  
    # first, make raster vrts of FORCE output that can then be used as extent raster for IACS rasterization
    force_path = f'{FORCE_folder}{state_folder}/{year}/'
    
    reduced_files = reduce_forceTSA_output_to_validmonths(force_path, 3, 8)
    ordered_files = force_order_Colors_for_VRT(reduced_files, colorL, [f'MONTH-{d:02d}' for d in range(3,9,1)])

    vrt_out = path_safe(f'{aux_vrt_path}{state_folder}/{year}/')

    if os.path.isdir(vrt_out):
        if len(getFilelist(f'{vrt_out}', '.vrt', deep=True)) > 0:
            print('VRT seems to be already computated, probably to create masks based on IACS')
        else:
            force_to_vrt(reduced_files, ordered_files, vrt_out, True, bandnames=colorL)
    else:
        os.makedirs(vrt_out)
        force_to_vrt(reduced_files, ordered_files, vrt_out, True, bandnames=colorL)

    vrt_cube_path = [file for file in getFilelist(vrt_out, '.vrt', deep=True) if 'Cube' in file][0]

    fields_path = f'{temp_folder}{state}/{state}_{year}_Fields.tif'
    polyon_lines_path = f'{temp_folder}{state}/{state}_{year}_lines.gpkg'
    borders_path = f'{temp_folder}{state}/{state}_{year}_rasterlines.tif'
    fieldsID_path = f'{temp_folder}{state}/{state}_{year}_Field_IDs.tif'

    # 1. rasterize fields (polygons) All_touch = False
    make_crop_mask(path_to_polygon=path,
                path_to_extent_raster=vrt_cube_path,
                path_to_mask_out=path_safe(fields_path),
                all_touch=False, 
                categories=EXCLUDE_LIST,
                category_col=state_exclude_column) 
    print('fields rasterized')

    # 2 .rasterize borders (after polygons to lines) All_touch = True
    polygons_to_lines(path_to_polygon=path,
                    path_to_lines_out=polyon_lines_path,
                    categories=EXCLUDE_LIST,
                    category_col=state_exclude_column)

    # rasterize lines
    rasterize_lines(path_to_lines=polyon_lines_path, 
                    path_to_extent_raster=vrt_cube_path, 
                    path_to_rasterlines_out=borders_path,
                    all_touch=True,
                    dilate=True)
    print('field borders rasterized')

    # 3. get field IDs 
    make_crop_mask(path_to_polygon=path,
                path_to_extent_raster=vrt_cube_path,
                path_to_mask_out=fieldsID_path,
                all_touch=False, 
                categories=EXCLUDE_LIST,
                category_col=state_exclude_column,
                burn_col='id_0') 
    print('unique fieldIDs rasterized')
    
    # 4. distance to border raster
    # as calculating the distance layer for the entire state is computational-wise too expensive, we cut image first in chips

    dist_path = f'{temp_folder}{state}/{state}_{year}_distance_to_border.vrt'
    if os.path.exists(dist_path):
        print(f'{dist_path} already created')
        continue

    vrt_ds = gdal.Open(fieldsID_path)
    geoTF = vrt_ds.GetGeoTransform()
    prj = vrt_ds.GetProjection()

    row_col_ind = get_row_col_indices(chip_size, 0, vrt_ds.RasterYSize, vrt_ds.RasterXSize)
    row_start = row_col_ind[0]
    row_end   = row_col_ind[1]
    col_start = row_col_ind[2]
    col_end   = row_col_ind[3]

    arr_IDs = vrt_ds.GetRasterBand(1).ReadAsArray()

    for i in range(len(row_end)):
        for j in range(len(col_end)):

            arr_dist = polygon_distance_normalized(arr_IDs[row_start[i]:row_end[i], col_start[j]:col_end[j]])
            distance_path = f'{temp_folder}{state}/{state}_{year}_rs{row_start[i]}_cs{col_start[j]}_distance_to_border.tif'
            # export labels as tif chips
            out_ds = gtiff_driver.Create(path_safe(distance_path), int(chip_size), int(chip_size), 1, gdal.GDT_Float32)
            # change the Geotransform for each chip
            geotf = list(geoTF)
            # get column and rows from filenames
            geotf[0] = geotf[0] + geotf[1] * col_start[j]
            geotf[3] = geotf[3] + geotf[5] * row_start[i]
     
            
            out_ds.SetGeoTransform(tuple(geotf))
            out_ds.SetProjection(prj)

            out_ds.GetRasterBand(1).WriteArray(arr_dist)
            del out_ds
    
    # create a vrt for distance layer
    vrt = gdal.BuildVRT(dist_path, [file for file in getFilelist(f'{temp_folder}{state}/', '.tif') if all(substr in file for substr in ['rs', 'cs'])])
    vrt = None
    print('distance calculated rasterized')


# now, for all stated states, IACS data are rasterized into binary Field Mask, Field Border Mask (dilated) and Distance to Border (normalized)


# ################################ 5. get the number of samples to draw, stratified against state size, field and border fraction and number of fields

# band_lkp = {0:'Field', 1:'Border',  2:'Distance', 3:'FieldIDS'}
# bands = len(band_lkp)

# # initialize result lists
# result = {
#     'state': [],
#     'year': [],
#     'band': [],
#     'row_start': [],
#     'row_end': [],
#     'col_start': [],
#     'col_end': [],
#     'PixelCount': [],
#     'Kill': []
# }

# for state, year in zip(states, years):
#     print(state)
#     # load the rasterized IACS data and stack it
#     fields_path = f'{temp_folder}{state}/{state}_{year}_Fields.tif'
#     borders_path = f'{temp_folder}{state}/{state}_{year}_rasterlines.tif'
#     dist_path = f'{temp_folder}{state}/{state}_{year}_distance_to_border.vrt'
#     fieldsID_path = f'{temp_folder}{state}/{state}_{year}_Field_IDs.tif'
 
#     arr_stack = stackReader(stack_tifs([fields_path, borders_path, dist_path, fieldsID_path], d_type=gdal.GDT_Float32))
#     row_col_ind =get_row_col_indices(chip_size, 0, arr_stack.shape[0], arr_stack.shape[1])
#     row_start = row_col_ind[0]
#     row_end   = row_col_ind[1]
#     col_start = row_col_ind[2]
#     col_end   = row_col_ind[3]

#     # keep track of outer edges of rasterized fields (as they are prone to arbitrary field vs masked values!!!)
#     dummy_box = np.zeros((len(row_start), len(col_start)))
#     ind_box = dummy_box.copy()

#     for i in range(len(row_end)):
#         for j in range(len(col_end)):
#             for band in range(bands):

#                 if band == 2 or band == 3:
#                     continue # not sure what should be stratified here --> too arbitrary??

#                 sub = arr_stack[row_start[i]:row_end[i], col_start[j]:col_end[j],band]
#                 # take the sum and append
#                 result['state'].append(state)
#                 result['year'].append(year)
#                 result['band'].append(band_lkp[band])
#                 result['row_start'].append(row_start[i])
#                 result['row_end'].append(row_end[i])
#                 result['col_start'].append(col_start[j])
#                 result['col_end'].append(col_end[j])

#                 if band ==1 and np.count_nonzero(sub) > 0:
#                     dummy_box[i, j] = 1

#                 # make a mask for all values that are not 0
#                 if band == 3:
#                     result['PixelCount'].append(len(np.unique(sub))-1) # substract -10000
#                 else:
#                     result['PixelCount'].append(np.count_nonzero(sub))

#     # find outer edge chips
#     for rowx in range(dummy_box.shape[0]):
#         for colx in range(dummy_box.shape[1]):
#             if dummy_box[rowx, colx] == 0:
#                 ind_box[rowx, colx] = 1
#             elif dummy_box[rowx, colx] != 0:
#                 ind_box[rowx, colx] = 1
#                 break

#     for colx in range(dummy_box.shape[1]):
#         for rowx in range(dummy_box.shape[0]):
#             if dummy_box[rowx, colx] == 0:
#                 ind_box[rowx, colx] = 1
#             elif dummy_box[rowx, colx] != 0:
#                 ind_box[rowx, colx] = 1
#                 break

#     for rowx in range(dummy_box.shape[0]):
#         for colx in range(dummy_box.shape[1]-1,-1,-1):
#             if dummy_box[rowx, colx] == 0:
#                 ind_box[rowx, colx] = 1
#             elif dummy_box[rowx, colx] != 0:
#                 ind_box[rowx, colx] = 1
#                 break

#     for colx in range(dummy_box.shape[1]):
#         for rowx in range(dummy_box.shape[0]-1,-1,-1):
#             if dummy_box[rowx, colx] == 0:
#                 ind_box[rowx, colx] = 1
#             elif dummy_box[rowx, colx] != 0:
#                 ind_box[rowx, colx] = 1
#                 break

#     for i in range(len(row_end)):
#         for j in range(len(col_end)):
#             for band in range(bands-2):
#                 if ind_box[i, j] ==1:
#                     result['Kill'].append(1)
#                 else:
#                     result['Kill'].append(0)


# # Convert results dict to DataFrame
# df = pd.DataFrame(result)

# # cut-off outer edge chips
# df = df[df["Kill"] == 0].copy()


# ### get a list of row col combis that only result in NA image pixel (Sentinel-2 vrts) as they are out of bounds
# df_mask = []

# for (state, year), group in df.groupby(['state', 'year']):
#     arr = stackReader([file for file in getFilelist(f'{aux_vrt_path}{state_folders[states.index(state)]}/{year}/', '.vrt', deep=True) if '_Cube' in file][0])
#     sub_df_0 = df[(df.state == state) & (df.year == year)]

#     mask = sub_df_0.apply(
#     lambda r: np.any(
#         arr[r.row_start:r.row_end, r.col_start:r.col_end, ...] < 0 |
#         np.isnan(arr[r.row_start:r.row_end, r.col_start:r.col_end, ...]) #== -9999
#     ),
#     axis=1 # makes it row-wise
#     )

#     df_mask.append(sub_df_0[~mask].copy())

# df_masked = pd.concat(df_mask)
# df_nonZero = df_masked[df_masked['PixelCount'] > min_pix_thresh]

# state_sizes = df_nonZero['state'].value_counts()
# samples_per_state = np.ceil(state_sizes / state_sizes.sum() * Total_number_of_samples).astype(int)

# # Create percentiles at 10% intervals
# percentiles = np.arange(0, 101, 20)  # 0, 10, 20, ..., 100


# colkeys = ['state', 'row_start', 'row_end', 'col_start', 'col_end']

# for stati in df_nonZero['state'].unique():

#     samples_per_bin = np.ceil((samples_per_state[stati]/ (len(percentiles) - 1)) / 2).astype(int)
    
#     for band in df_nonZero['band'].unique():
     
#         group = df_nonZero[(df_nonZero['band'] == band) & (df_nonZero['state'] == stati)]
        
#         # Compute percentile thresholds for this band's PixelCount
#         if group is None or group.empty:
#             print(f"empty group, no samples for band {band} possible in {stati}")
#         else:
#             thresholds = np.percentile(group['PixelCount'], percentiles)
        
#         # Iterate over percentile *ranges* (0–10%, 10–20%, etc.)
#         for k in range(len(thresholds) - 1):
#             lower, upper = thresholds[k], thresholds[k + 1]
#             subset = group[(group['PixelCount'] >= lower) & (group['PixelCount'] < upper)]

#             if subset.empty:
#                 continue
            
#             # Draw random samples from this range
#             n_take = min(samples_per_bin, len(subset))
#             sampled_rows = subset.sample(n_take, random_state=seed)
        
#             # Drop from df_nonZero all rows whose block exists in sampled_rows
#             df_nonZero = (
#             df_nonZero
#             .merge(sampled_rows[colkeys].drop_duplicates(), on=colkeys, how='left', indicator=True)
#             .query('_merge == "left_only"')
#             .drop(columns=['_merge'])
#             )

# # 6. Combine all sampled data and add 5% of complete 0 raster
# sample_train_df = pd.concat(samples_train).reset_index(drop=True)

# write away the samples for checking

# samp0 = int((len(sample_train_df) + len(sample_test_df)) / 80)

# # filter the oroginal df_masked for a state and rol col combi, where field and border are both 0
# colkeys_state = colkeys + ['state']
# df_all0 = df_masked.groupby(colkeys_state).filter(lambda g: (g[g['band'].isin(['Field', 'Border'])]['PixelCount'] == 0).all())
# df_all0_unique = df_all0[df_all0['band'] == 'Field'] # drop duplicates for state row col combi


# # get distribution of empty chips per state
# state_sizes0 = df_all0_unique['state'].value_counts()
# samples0_per_state = np.ceil(state_sizes0 / state_sizes0.sum() * samp0).astype(int)

# # draw samples accordingly per state
# samples0_train, samples0_test = [], []
# for statX in df_all0_unique['state'].unique():
#     group = df_all0_unique[df_all0_unique['state'] == statX]
#     nn_take = min(samples0_per_state[statX], len(group))
#     sampled_rows0 = group.sample(nn_take, random_state=seed)

#     # Make a reproducible random split
#     indices = list(range(len(sampled_rows0)))
#     random.shuffle(indices)
#     split_idx = int(len(indices) * 0.85)
#     samples0_train.append(sampled_rows0.iloc[indices[:split_idx]])
#     samples0_test.append(sampled_rows0.iloc[indices[split_idx:]])

# # merge samples of empty chips with those that contain data
# sample_block_train = pd.concat([sample_train_df, pd.concat(samples0_train)]).sort_values(by='state')

# write away the samples for checking

# # label and img chips creation

# for state, state_folder, year in zip(states, state_folders, years):

#     # get geo information
#     print(state)
#     fields_path = f'{temp_folder}{state}/{state}_{year}_Fields.tif'
#     borders_path = f'{temp_folder}{state}/{state}_{year}_rasterlines_touch_true.tif'
#     dist_path = f'{temp_folder}{state}/{state}_{year}_distance_to_border.vrt'
#     fieldsID_path = f'{temp_folder}{state}/{state}_{year}_Field_IDs.tif'
 
#     arr_stack = stackReader(stack_tifs([fields_path, borders_path, dist_path, fieldsID_path], d_type=gdal.GDT_Float32))
#     row_col_ind =get_row_col_indices(chip_size, 0, arr_stack.shape[0], arr_stack.shape[1])
#     row_start = row_col_ind[0]
#     row_end   = row_col_ind[1]
#     col_start = row_col_ind[2]
#     col_end   = row_col_ind[3]

#     arr_stack[:,:,3][arr_stack[:,:,3] == 0] = -10000 # makes the background of uniqueIDs same as in ai4bound
#     vrt_ds = gdal.Open(f'{temp_folder}{state}/{state}_{year}_Fields.tif')
#     vrt_arr = vrt_ds.GetRasterBand(1).ReadAsArray()
#     geoTF = vrt_ds.GetGeoTransform()
#     prj = vrt_ds.GetProjection()

#     force_arr = loadVRTintoNumpyAI4(f"{aux_vrt_path}{state_folder}/{year}/{dirfinder(f'{aux_vrt_path}{state_folder}/{year}/')[0]}",
#                                     applyNormalizer=False)

#     # export training set
#     for idx, row in sample_block_train.iterrows():#sample_df[sample_df['band'] == band_lkp[band]].iterrows():
        
#         if row['state'] != state:
#             continue
#         # export labels as tif chips
#         out_ds = gtiff_driver.Create(
#             path_safe(
#                 f"{out_folder}label/{state_folder}/{year}/{state_folder}_{year}_rowstart_{row['row_start']:04d}_colstart_{row['col_start']:04d}.tif"),
#                                         int(chip_size), int(chip_size), 4, gdal.GDT_Float32)
#         # change the Geotransform for each chip
#         geotf = list(geoTF)
#         # get column and rows from filenames
#         geotf[0] = geotf[0] + geotf[1] * row['col_start']
#         geotf[3] = geotf[3] + geotf[5] * row['row_start']
#         #print(f'X:{geoTF[0]}  Y:{geoTF[3]}  AT {file}')
        
#         out_ds.SetGeoTransform(tuple(geotf))
#         out_ds.SetProjection(prj)

#         for bandx in range(arr_stack.shape[-1]):
#             out_ds.GetRasterBand(bandx + 1).WriteArray(arr_stack[row['row_start']:row['row_end'], row['col_start']:row['col_end'],bandx])
#             out_ds.GetRasterBand(bandx + 1).SetNoDataValue(-10000)
#         del out_ds

#         # make Sentinel-2 chips as .nc
#         force_sub = force_arr[:,:,row['row_start']:row['row_end'], row['col_start']:row['col_end']].copy()
#         # set values below 1 to -9999
#         force_sub[force_sub < 0] = -9999

#         time = np.arange(6)
#         x = geotf[0] + np.arange(chip_size) * geotf[1]
#         y = geotf[3] + np.arange(chip_size) * geotf[5]
 
#         data_vars = {colorL[i]: (("time", "y", "x"), force_sub[i]) for i in range(len(colorL))}

#         ds = xr.Dataset(data_vars=data_vars, coords={"time": time, "x": x, "y": y})

#         srs = osr.SpatialReference()
#         srs.ImportFromWkt(vrt_ds.GetProjection())
#         epsg_code = srs.GetAttrValue('AUTHORITY', 1)

#         ds.rio.write_crs(f'EPSG:{epsg_code}', inplace=True)

#         encoding = {band: {"zlib": True, "_FillValue": -9999} for band in colorL}
#         ds.to_netcdf(
#             path_safe(f"{out_folder}img/{state_folder}/{year}/{state_folder}_{year}_rowstart_{row['row_start']:04d}_colstart_{row['col_start']:04d}.nc"),
#             encoding=encoding)