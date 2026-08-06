import sys
origin = '/workspace/'
sys.path.append('/media/')

from tqdm import tqdm
from FieldWaterUseTools.FuncBox.Misc import path_safe, dirfinder, get_row_col_indices, checkPath, getFilelist,\
     stackReader, warp_np_to_reference
from FieldWaterUseTools.FuncBox.FieldFuncis import *
from FieldWaterUseTools.FuncBox.ForceFuncis import *
from FieldWaterUseTools.FuncBox.DICT_LIST import VALID_AGRO_VALUES
from shapely.geometry import box
from concurrent.futures import ProcessPoolExecutor
import geopandas as gpd
import math
import zipfile

####################################################### Prepare
model_name = 'model_state_FromScratch_IACS_dilate_True_BorderEdgeCutted_RGB_NDVI_exclude_True_with_overlap_47'
model_path = f"{origin}fields/03_Output/models/{model_name}.pth"
year = 2025
colorList = ['BLUE', 'GREEN', 'RED', 'BROADNIR']

# make vrts from force outputs for easier processing
state = 'GERMANY'

vrt_out = path_safe(f'{origin}fields/Auxiliary/vrt/{state}/{year}/')
reduced_files = reduce_forceTSA_output_to_validmonths(f'{origin}force/output/{state}/{year}/', 3, 8)
ordered_files = force_order_Colors_for_VRT(reduced_files, colorList, [f'MONTH-{d:02d}' for d in range(3,9,1)])

if os.path.isdir(vrt_out):
    if len(getFilelist(f'{vrt_out}', '.vrt', deep=True)) > 0:
        print(f'VRT seems to be already computated for {year}, probably to create masks based on IACS')
    else:
        force_to_vrt(reduced_files, ordered_files, vrt_out, False, bandnames=colorList)
else:
    os.makedirs(vrt_out)
    force_to_vrt(reduced_files, ordered_files, vrt_out, False, bandnames=colorList)


# set folder
predict_master_folder = path_safe(f"{origin}fields/04_Predictions/{state}/{model_name.split('_state_')[-1]}/{year}/")
vrt_Folder = f"{origin}/fields/Auxiliary/vrt/{state}/{year}/{dirfinder(f'{origin}/fields/Auxiliary/vrt/{state}/{year}/')[0]}"
vrtFiles = [file for file in getFilelist(vrt_Folder, '.vrt') if 'Cube' not in file]
vrtFiles = sortListwithOtherlist([int(vrt.split('_')[-1].split('.')[0]) for vrt in vrtFiles], vrtFiles)[-1]

vrt_ds = gdal.Open(vrtFiles[0])

# set tiling scheme and chip size on which prediction will be undertaken
chipsize = 128*2 # 5 is the maximum with GPU in basement
overlap  = 20

row_col_ind = get_row_col_indices(chipsize, overlap, vrt_ds.RasterYSize, vrt_ds.RasterXSize)

####################################################### Predict
print('start prediction')
predicted_chips_list = predict_on_GPU_without_preload(model_path, row_col_ind, vrtFiles, 
                                      temp_path=f'{predict_master_folder}temp/')


# export the predicted chips (masked and not masked)
print('predicted - write away to temp')
with open(f'{predict_master_folder}temp/preds.pkl', 'rb') as f:
    predicted_chips_list = pickle.load(f)

export_GPU_predictions(predicted_chips_list, 
                    'no mask', 
                    vrt_Folder,
                    row_col_ind, 
                    path_safe(f'{predict_master_folder}chips_folder/'),
                    chipsize, overlap)
print('create vrt')
# # make vrt of predicted image chips
for chip in dirfinder(f'{predict_master_folder}chips_folder/'):
    predicted_chips_to_vrt(f'{predict_master_folder}chips_folder/', chip,  chipsize, overlap,
                        path_safe(f'{predict_master_folder}vrt/'), pyramids=True)
    


print("create thuenen mask and export")
### open prediction and mask it with THuenen crop type map
# get paths and cut thuenen to prediction
thuenen_path = [file for file in getFilelist(f"{origin}et/Auxiliary/landcover/thuenen", '.tif') if f"_{year}_" in file][0]
vrt_path = getFilelist(f"{predict_master_folder}vrt/", ".vrt")[0]
subset_mask_to_prediction_extent(path_reference_mask=thuenen_path, path_to_prediction_vrt=vrt_path, area='delete_after_usage')

# load mask
mask_path = [file for file in getFilelist(f"{origin}et/Auxiliary/landcover/thuenen", '.tif') if 'delete_after_usage' in file and f"_{year}_" in file][0]
# open prediction and thuenen and mask prediction
mask_ds = gdal.Open(mask_path)
mask_arr = mask_ds.GetRasterBand(1).ReadAsArray()
mask = np.where(mask_arr == 0, 0, 1)

# mask and export
predL = []
pred_ds = gdal.Open(vrt_path)
bds = pred_ds.RasterCount
for band in range(bds):
    pred_arr = pred_ds.GetRasterBand(band + 1).ReadAsArray()
    predL.append(pred_arr * mask)

creation_options = [
    'COMPRESS=ZSTD',
    'PREDICTOR=3',       # float-optimized predictor
    'ZSTD_LEVEL=9',
    'TILED=YES',
    'BLOCKXSIZE=512',
    'BLOCKYSIZE=512',
    'BIGTIFF=YES',       # required for files > 4GB
]


out_ds = gdal.GetDriverByName('GTiff').Create(f"{predict_master_folder}vrt/Masked_THUENEN_CTM_{year}.tif", 
                                              mask.shape[1], mask.shape[0], bds, pred_ds.GetRasterBand(1).DataType,
                                              options=creation_options)
out_ds.SetGeoTransform(pred_ds.GetGeoTransform())
out_ds.SetProjection(pred_ds.GetProjection())
for band in range(bds):
    out_ds.GetRasterBand(band + 1).WriteArray(predL[band])
del out_ds


# mask chips
print("mask chips")
chips_folder = f"{predict_master_folder}chips_folder/unmasked_chips/"
masked_folder = path_safe(f"{predict_master_folder}chips_folder/masked_chips/")
files = getFilelist(chips_folder, '.tif')
mask_ds = gdal.Open(f"{predict_master_folder}vrt/Masked_THUENEN_CTM_{year}.tif")

conti = []
for file in files:
    # load masked stack
    chip = mask_ds.GetRasterBand(1).ReadAsArray(
        xoff=int(file.split('X_')[-1].split('_')[0]),
        yoff=int(file.split('X_')[-1].split('_')[2].split('.')[0]),
        win_xsize=236,  
        win_ysize=236 
    )

    if np.nansum(chip) > 0:

        with rasterio.open(file) as src:
            bounds = src.bounds  # left, bottom, right, top
            geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            crs = src.crs

            conti.append({
                "filename": os.path.basename(file),
                "geometry": geom,
                "crs_used": str(crs)
            })

            chip2 = mask_ds.GetRasterBand(2).ReadAsArray(
                xoff=int(file.split('X_')[-1].split('_')[0]),
                yoff=int(file.split('X_')[-1].split('_')[2].split('.')[0]),
                win_xsize=236,  
                win_ysize=236 
            )

            chip3 = mask_ds.GetRasterBand(3).ReadAsArray(
                xoff=int(file.split('X_')[-1].split('_')[0]),
                yoff=int(file.split('X_')[-1].split('_')[2].split('.')[0]),
                win_xsize=236,  
                win_ysize=236 
            )

            stack = np.stack([chip, chip2, chip3], axis=0)

            with rasterio.open(
                f"{masked_folder}chips_masked256{file.split('_unmasked256')[-1]}",
                "w",
                driver="GTiff",
                height=stack.shape[1],
                width=stack.shape[2],
                count=stack.shape[0],
                dtype=stack.dtype,
                crs=crs,          # z.B. von einem Referenz-Datensatz: ref_ds.crs
                transform=src.transform  # z.B. ref_ds.transform
            ) as dst:
                dst.write(stack)

# make vrt of masked chips
print('vrt of masked chips')
predicted_chips_to_vrt(f"{predict_master_folder}chips_folder/", 'masked_chips', 256, 20, path_safe(f"{predict_master_folder}vrt/"), pyramids=True)

# zip masked chips
mfiles = getFilelist(masked_folder, '.tif')
print(len(mfiles))
n = 8
chunk_size = math.ceil(len(mfiles) / n)
chunks = [mfiles[i:i + chunk_size] for i in range(0, len(mfiles), chunk_size)]


def zip_chunk(args):
    idx, chunk, masked_folder = args

    zip_name = os.path.join(masked_folder, f"masked_{idx}.zip")

    with zipfile.ZipFile(
        zip_name,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:
        for tif_path in chunk:
            zf.write(tif_path, arcname=os.path.basename(tif_path))

print('zipping masked chips')
with ProcessPoolExecutor(max_workers=30) as executor:
    executor.map(
        zip_chunk,
        [(idx, chunk, masked_folder) for idx, chunk in enumerate(chunks)]
    )


# export vector of chips
print('exporting shapes masked chips')
gdf = gpd.GeoDataFrame(conti, geometry="geometry", crs=conti[0]["crs_used"])
gdf.to_file(f"{predict_master_folder}{year}_grid_tiles.gpkg", driver="GPKG", layer="tiles")

# delete thuenen cut to prediction
RasterKiller(mask_path)