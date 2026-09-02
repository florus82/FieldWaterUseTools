import os
import sys
import time
import requests
import openeo
import cdsapi

# Dynamic dataset needed are ERA-5 variables, Thuenen crop-type maps, Sentinel-2 and Sentinel-3 dataset

origin = '/workspace/'
sys.path.append('/media/')

from Misc import getFilelist, path_safe, slash_checker, download_thuenen_cropTypes # FieldWaterUseTools.FuncBox.
from DICT_LIST import REAL_INT_TO_MONTH # FieldWaterUseTools.FuncBox.

# set year and month for which to obtain data

YEAR = 2026
MONTH = 5

# set masterpath for stored data
storPath_master = slash_checker(path_safe(f"{origin}etZ_REPO_TEST/"))#'/place/to/store/porducts/')
storPath_S2 = slash_checker(path_safe(f"{storPath_master}{YEAR}/{MONTH:02d}/S2/"))
storPath_S3 = slash_checker(path_safe(f"{storPath_master}{YEAR}/{MONTH:02d}/S3/"))
storPath_ERA5 = slash_checker(path_safe(f"{storPath_master}{YEAR}/{MONTH:02d}/ERA5/"))
storPath_GeoPot = slash_checker(path_safe(f"{storPath_ERA5}GEOPOT/"))
storPath_Thuen = slash_checker(path_safe(f"{storPath_master}{YEAR}/Thuenen/"))




# ########################################################################## ERA5

# client = cdsapi.Client()

# dataset = "reanalysis-era5-single-levels"

# variables = [             
#         "2m_dewpoint_temperature", 
#         "2m_temperature",
#         "surface_pressure",
#         "100m_u_component_of_wind",
#         "100m_v_component_of_wind",
#         "total_column_water_vapour",
#         "surface_solar_radiation_downward_clear_sky"]

# for variable in variables:
    
#     varPath = path_safe(f"{storPath_ERA5}grib/{variable}")
#     storPath = f"{varPath}/{variable}_{YEAR}_{REAL_INT_TO_MONTH[MONTH]}.grib"

#     if os.path.exists(storPath):
#         pass
#     else:
#         try:
#             request = {
#                 "product_type": ["reanalysis"],
#                 "variable": [variable],
#                 "year": [YEAR],
#                 "month": [MONTH],
#                 "day": [
#                     "01", "02", "03",
#                     "04", "05", "06",
#                     "07", "08", "09",
#                     "10", "11", "12",
#                     "13", "14", "15",
#                     "16", "17", "18",
#                     "19", "20", "21",
#                     "22", "23", "24",
#                     "25", "26", "27",
#                     "28", "29", "30",
#                     "31"
#                 ],
#                 "time": [
#                     "00:00", "01:00", "02:00",
#                     "03:00", "04:00", "05:00",
#                     "06:00", "07:00", "08:00",
#                     "09:00", "10:00", "11:00",
#                     "12:00", "13:00", "14:00",
#                     "15:00", "16:00", "17:00",
#                     "18:00", "19:00", "20:00",
#                     "21:00", "22:00", "23:00"
#                 ],
#                 "data_format": "grib",
#                 "download_format": "unarchived",
#                 "area": [56, 5, 47, 16]
#             }

#             target = storPath

#             client.retrieve(dataset, request, target)
        
#         except Exception as e:
#             print(e)
#             t = time.localtime()
#             ti = time.strftime("%H:%M:%S", t)
#             print(f"thrown at {ti}")
#             continue

# if os.path.exists(path_safe(path_to_geopot_raw)):
#     pass
# else:
#     try:
#         request = {
#             "product_type": ["reanalysis"],
#             "variable": ["geopotential"],
#             "year": [2020],
#             "month": [MONTH],
#             "day": ["01"],
#             "time": ["13:00"],
#             "data_format": "grib",
#             "download_format": "unarchived",
#             "area": [56, 5, 47, 16]
#         }


#         target = path_to_geopot_raw

#         client.retrieve(dataset, request, target)

#     except Exception as e:
#         print(e)
#         t = time.localtime()
#         ti = time.strftime("%H:%M:%S", t)
#         print(f"thrown at {ti}")


########################################################################## Thuenen maps agriculture
# if f"{path_to_Thuenen}"




################################################################### Sentinel-3 compositing
#set paths
path_to_S3_composites = path_safe(f"{storPath_master}Sentinel3/")
LST_path = f"{path_to_S3_composites}LST/"
VZA_path = f"{path_to_S3_composites}VZA/"
VAA_path = f"{path_to_S3_composites}VAA/"

LST_minVZA_path = path_safe(f"{LST_path}minVZA/{YEAR}/")
LST_maxLST_path = path_safe(f"{LST_path}maxLST/{YEAR}/")
VZA_minVZA_path = path_safe(f"{VZA_path}minVZA/{YEAR}/")
VZA_maxLST_path = path_safe(f"{VZA_path}maxLST/{YEAR}/")
VAA_minVZA_path = path_safe(f"{VAA_path}minVZA/{YEAR}/")
VAA_maxLST_path = path_safe(f"{VAA_path}maxLST/{YEAR}/")

AcqTime_stor_path = path_safe(f"{path_to_S3_composites}Acq_Time/{YEAR}/")

# Define bounding box and time frame for Germany for S3 compositing
bbox = [5.592041, 47.129951, 15.26001, 55.09723]  # Germany
bbox_shape = box(*bbox)
wkt_bbox = bbox_shape.wkt

start = f"{YEAR}-{MONTH:02d}-01"
if is_leap_year(YEAR):
   end = f"{YEAR}-{MONTH:02d}-{DAYCOUNT_LEAP[MONTH-1]}"
else:
    end = f"{YEAR}-{MONTH:02d}-{DAYCOUNT_NOLEAP[MONTH-1]}"

# read AZURE CLOUD KEY from file in personal folder
with open ('/data/Aldhani/users/potzschf/rss.txt') as f:
    STORAGE_ACCOUNT = f.readline().strip()
    SAS_KEY = f.readline().strip()


# Setup Azure filesystem and DuckDB connection
azure_blob_fileservice = fsspec.filesystem("abfs", account_name=STORAGE_ACCOUNT, sas_token=SAS_KEY)
con = duckdb.connect()
con.execute("INSTALL spatial; LOAD spatial;")
con.register_filesystem(azure_blob_fileservice)
stac_geoparquet = "abfs://stac/catalog.parquet"

# Execute query
sql_query = get_query(start, end, wkt_bbox, stac_geoparquet)
db = con.query(sql_query)
table = db.fetch_arrow_table()

# Convert to STAC items and add SAS tokens
items = [
    add_sas_token(pystac.Item.from_dict(item_dict), SAS_KEY)
    for item_dict in stac_table_to_items(table)
]

print(f"Found {len(items)} STAC items for {MONTH}")

# Define bands to load
bands = ["lst", "bayes", "confidence"]# , "vza", "vaa"] # vza & vaa placeholder for viewing zenith/azimuth angles

# Load data using ODC-STAC
ds = odc.stac.load(
    items,
    bbox=bbox,
    chunks={"x": 512, "y": 512},
)

# Remove pixels where Bayes cloud probability indicates clouds (value = 2)
ds["lst"] = ds["lst"].where(ds["bayes"] != 2, np.nan)

# Remove pixels with low confidence (confidence < 16384 or >= 32678 indicates good quality)
ds["lst"] = ds["lst"].where((ds["confidence"] < 16384) | (ds["confidence"] >= 32678), np.nan)

# get lst values
lst_np = ds["lst"].values
dat_LST = lst_np.transpose(1,2,0)
dat_LST[dat_LST<273.15] = np.nan # LST_MASKING check!
# vza_np = ds['vza'] # swap with real band name"
# vaa_np = ds['vaa'] # swap with real band name"
# dat_VZA = vza_np.transpose(1,2,0)
# dat_VAA = vaa_np.transpose(1,2,0)

# get acquisition time of scenes
time_np = ds['time'].data  # already a NumPy array of datetime64[ns] or [us]
accDateTimes = time_np.astype('datetime64[ns]')
df = pd.Series(accDateTimes)
counts_per_day = df.dt.floor("D").value_counts().sort_index()
# make iterables from counts per day that catch starting and ending indices to subset all obs per day
cumulative_day_counts_end = np.asarray(np.cumsum(counts_per_day))
cumulative_day_counts_start = np.insert(cumulative_day_counts_end, 0 ,0)

# check air temperature (2m ERA5)
dat_2mT, time_2mT = stackReader(path_to_airTemp_S3, bands=True)

#### get ERA5 AirTemp (interpolated from both modelled values that are closest to LST)
bands_low = []
minutes = [] # get the minutes to interpolate ERA5 temp values to the exact minute of LST acquisition
for accDT in accDateTimes: # search for each LST observation 
    accDT_utc = pd.Timestamp(accDT, tz="UTC")
    for count, air_time in enumerate(time_2mT): # the two neighbouting ERA5 air temp values
        if accDT_utc.floor('h') == pd.Timestamp(air_time, tz="UTC"): # this will get the hourly value before the acquisition
            bands_low.append(count)
            minutes.append(accDT_utc.minute)
            break

bands_up = [band + 1 for band in bands_low]# this get the hourly value after the acquisition

# interpolate to the minute of observation
air_temp_intpol = dat_2mT[:,:,bands_low] - (dat_2mT[:,:,bands_low] - dat_2mT[:,:,bands_up]) * (np.array(minutes, dtype=np.float32) / 60).reshape(1,1,-1)

# apply air threshold
dat_LST = np.where((dat_LST - air_temp_intpol) < -2, np.nan, dat_LST)

# now get composites (minVZA, maxLST)
minVZA_LST = [] # collects 2D numpy arrays with masked LST values from minVZA compositiing
maxLST_LST = [] # collects 2D numpy arrays with masked LST values from maxLST compositiing
minVZA_VZA = [] # collects 2D numpy arrays with masked VZA values from minVZA compositiing
maxLST_VZA = [] # collects 2D numpy arrays with masked VZA values from maxLST compositiing
minVZA_VAA = [] # collects 2D numpy arrays with masked VAA values from minVZA compositiing
maxLST_VAA = [] # collects 2D numpy arrays with masked VAA values from maxLST compositiing

minACQL = [] # collect acquisition times of minVZA pixel
minACQL_read = [] # collect readable acquisition times of minVZA pixel
maxACQL = [] # collect acquisition times of maxLST pixel
maxACQL_read = [] # collect readable acquisition times of maxLST pixel 

doyL = [] # for band names when exporting

for l in range(len(counts_per_day)):

    ################## LST values
    # Select the slices for the day:
    LST_slice = dat_LST[:, :, cumulative_day_counts_start[l]:cumulative_day_counts_end[l]]  # shape (X,Y,Z)
    # VZA_slice = dat_VZA[:, :, cumulative_day_counts_start[l]:cumulative_day_counts_end[l]]  # shape (X,Y,Z)
    # VAA_slice = dat_VAA[:, :, cumulative_day_counts_start[l]:cumulative_day_counts_end[l]]  # shape (X,Y,Z)

    # Create mask where LST is valid and VZA < 45
    valid_mask = (~np.isnan(LST_slice)) # & (VZA_slice < 45)
    
    # For each (x,y), set VZA/LST invalid points to a large number so they don't become min
    # vza_for_minVZA = np.where(valid_mask, VZA_slice, np.inf)  # shape (X,Y,Z)
    lst_for_maxLST = np.where(valid_mask, LST_slice, -np.inf)  # shape (X,Y,Z)

    # Find index of minimal VZA/max LST along axis=2 (time/bands) for each pixel
    # minVZA_idx = np.argmin(vza_for_minVZA, axis=2)  # shape (X,Y)
    maxLST_idx = np.argmax(lst_for_maxLST, axis=2)  # shape (X,Y)

    # Now use advanced indexing to get the corresponding LST values:
    x_indices = np.arange(LST_slice.shape[0])[:, None]  # shape (X,1)
    y_indices = np.arange(LST_slice.shape[1])[None, :]  # shape (1,Y)

    # fill 
    # best_LST_minVZA = LST_slice[x_indices, y_indices, minVZA_idx]  # shape (X,Y)
    best_LST_maxLST = LST_slice[x_indices, y_indices, maxLST_idx]  # shape (X,Y)
    # best_VZA_minVZA = VZA_slice[x_indices, y_indices, minVZA_idx]  # shape (X,Y)
    # best_VZA_maxLST = VZA_slice[x_indices, y_indices, maxLST_idx]  # shape (X,Y)
    # best_VAA_minVZA = VAA_slice[x_indices, y_indices, minVZA_idx]  # shape (X,Y)
    # best_VAA_maxLST = VAA_slice[x_indices, y_indices, maxLST_idx]  # shape (X,Y)

    # Take care of all invalid pixel that might have sneaked in through np.argmin
    no_valid_points_minVZA = ~np.any(valid_mask, axis=2)  # shape (X,Y)
    no_valid_points_maxLST = ~np.any(valid_mask, axis=2)  # shape (X,Y)
    
    # best_LST_minVZA[no_valid_points_minVZA] = np.nan
    best_LST_maxLST[no_valid_points_maxLST] = np.nan

    # best_VZA_minVZA[no_valid_points_minVZA] = np.nan
    # best_VZA_maxLST[no_valid_points_maxLST] = np.nan

    # best_VAA_minVZA[no_valid_points_minVZA] = np.nan
    # best_VAA_maxLST[no_valid_points_maxLST] = np.nan

    # minVZA_LST.append(best_LST_minVZA) # * mask)
    maxLST_LST.append(best_LST_maxLST) # * mask)
    
    # minVZA_VZA.append(best_VZA_minVZA) # * mask)
    # maxLST_VZA.append(best_VZA_maxLST) # * mask)

    # minVZA_VAA.append(best_VAA_minVZA) # * mask)
    # maxLST_VAA.append(best_VAA_maxLST) # * mask)

    doyL.append(f'DOY_{l+1}')



    # ################# Time of observation of selected pixel --> needed for ERA5
    time_slice = df[cumulative_day_counts_start[l]:cumulative_day_counts_end[l]].values
    timestamp_array = np.tile(time_slice, dat_LST.shape[:2] + (1,)) # don’t repeat along the time axis, just preserve it (1,) 
    
    # acq_time = timestamp_array[x_indices, y_indices, minVZA_idx]  
    # acq_time_unix = acq_time.astype('datetime64[s]').astype(int) # convert back with pd.to_datetime(best_time_unix, unit='s')
    # acq_time_unix[no_valid_points_minVZA] = 0 # use 0 as na for export
    # minACQL.append(acq_time_unix) # * mask)
    
    acq_time = timestamp_array[x_indices, y_indices, maxLST_idx]  
    acq_time_unix = acq_time.astype('datetime64[s]').astype(int) # convert back with pd.to_datetime(best_time_unix, unit='s')
    acq_time_unix[no_valid_points_maxLST] = 0 # use 0 as na for export
    maxACQL.append(acq_time_unix) # * mask)

    # and also as readable tiffs --> just for debugging and visualization needed
    datetimes = time_slice.astype('datetime64[m]').astype('O')
    time_arr = np.array([int(dt.strftime("%H%M")) for dt in datetimes])
    timestamp_array_read = np.tile(time_arr, dat_LST.shape[:2] + (1,))
    
    # acq_time = timestamp_array_read[x_indices, y_indices, minVZA_idx]  
    # acq_time[no_valid_points_minVZA] = 0 # use 0 as na for export
    # minACQL_read.append(acq_time) # * mask)

    acq_time = timestamp_array_read[x_indices, y_indices, maxLST_idx]  
    acq_time[no_valid_points_maxLST] = 0 # use 0 as na for export
    maxACQL_read.append(acq_time) # * mask)


# ################## export minVZA LST composite
# # LST masked
# npTOdisk(arr=np.dstack(minVZA_LST), reference_path=path_to_airTemp_S3,
#             outPath=f"{LST_minVZA_path}Daily_LST_minVZA_{YEAR}_{REAL_INT_TO_MONTH[MONTH]}.tif", 
#             bands=len(minVZA_LST), bandnames=doyL, noData=0)

# # VZA masked
# npTOdisk(arr=np.dstack(minVZA_VZA), reference_path=path_to_airTemp_S3,
#         outPath=f"{VZA_minVZA_path}Daily_VZA_minVZA_{YEAR}_{REAL_INT_TO_MONTH[MONTH]}.tif", 
#         bands=len(minVZA_VZA), bandnames=doyL, noData=0)

# # VAA masked
# npTOdisk(arr=np.dstack(minVZA_VAA), reference_path=path_to_airTemp_S3,
#             outPath=f"{VAA_minVZA_path}Daily_VAA_minVZA_{YEAR}_{REAL_INT_TO_MONTH[MONTH]}.tif", 
#             bands=len(minVZA_VAA), bandnames=doyL, noData=0)

# # time
# npTOdisk(arr=np.dstack(minACQL), reference_path=path_to_airTemp_S3,
#             outPath=f"{AcqTime_stor_path}Daily_AcqTime_minVZA_{YEAR}_{REAL_INT_TO_MONTH[MONTH]}.tif",
#             bands=len(minACQL), bandnames=doyL, noData=0)

# npTOdisk(arr=np.dstack(minACQL_read), reference_path=path_to_airTemp_S3,
#             outPath=f"{AcqTime_stor_path}Daily_AcqTime_minVZA_{YEAR}_{REAL_INT_TO_MONTH[MONTH]}_readable.tif",
#             bands=len(minACQL), bandnames=doyL, noData=0)


################# export max LST composite
# LST masked
npTOdisk(arr=np.dstack(maxLST_LST), reference_path=path_to_airTemp_S3,
            outPath=f"{LST_maxLST_path}Daily_LST_maxLST_{YEAR}_{REAL_INT_TO_MONTH[MONTH]}.tif",
            bands=len(maxLST_LST), bandnames=doyL, noData=0)

# # VZA masked
# npTOdisk(arr=np.dstack(maxLST_VZA), reference_path=path_to_airTemp_S3,
#             outPath=f"{VZA_maxLST_path}Daily_VZA_maxLST_{YEAR}_{REAL_INT_TO_MONTH[MONTH]}.tif",
#             bands=len(maxLST_VZA), bandnames=doyL, noData=0)

# # VAA masked
# npTOdisk(arr=np.dstack(maxLST_VAA), reference_path=path_to_airTemp_S3,
#             outPath=f"{VAA_maxLST_path}Daily_VAA_maxLST_{YEAR}_{REAL_INT_TO_MONTH[MONTH]}.tif",
#             bands=len(maxLST_VAA), bandnames=doyL, noData=0)

# time
npTOdisk(arr=np.dstack(maxACQL), reference_path=path_to_airTemp_S3,
            outPath=f"{AcqTime_stor_path}Daily_AcqTime_maxLST_{YEAR}_{REAL_INT_TO_MONTH[MONTH]}.tif",
            bands=len(maxACQL), bandnames=doyL, noData=0)

npTOdisk(arr=np.dstack(maxACQL_read), reference_path=path_to_airTemp_S3,
            outPath=f"{AcqTime_stor_path}Daily_AcqTime_maxLST_{YEAR}_{REAL_INT_TO_MONTH[MONTH]}_readable.tif",
            bands=len(maxACQL), bandnames=doyL, noData=0)