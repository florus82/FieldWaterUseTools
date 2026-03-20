import os
import sys
from osgeo import gdal
import re
import xdem
import rasterio
import numpy as np
from pyproj import Transformer
from rasterio.windows import from_bounds

import sys
sys.path.append('/home/potzschf/repos/')
origin =  '/data/Aldhani/eoagritwin/'

from FieldWaterUseTools.FuncBox.Misc import getFilelist, path_safe
from FieldWaterUseTools.FuncBox.EvapFuncis import growingSeasonChecker, warp_ERA5_to_reference, warp_raster_to_reference
from FieldWaterUseTools.FuncBox.DICT_LIST import INT_TO_MONTH, STANDARD_ADIABAT
from FieldWaterUseTools.FuncBox.ForceFuncis import convertVRTpathsTOrelative


################################################################### DEM preprocessing
# set paths
storPath_master = '/place/to/store/porducts/' # f"{origin}et/test/"
path_to_S3_template = f"{storPath_master}templates/S3_template.tif"

storPath_DEM = f"{storPath_master}DEM/"
storPath_TILES = f"{storPath_DEM}FORCE_TILES/" # only works if DEM TILES from FORCE tiling scheme are stored there. Otherwise, raw tiles needed to be reprojected first
storPath_DEM_TILES = f"{storPath_TILES}DEM/"

DEM_GER_path = f"{storPath_DEM}vrt_and_derivates/DEM_GER_S2.vrt"
slope_path = f'{storPath_DEM}vrt_and_derivates/SLOPE_GER_S2.tif'
aspect_path = f'{storPath_DEM}vrt_and_derivates/ASPECT_GER_S2.tif'
lat_path = f"{storPath_DEM}vrt_and_derivates/LON_GER_S2.tif"
lon_path = f"{storPath_DEM}vrt_and_derivates/LAT_GER_S2.tif"

if os.path.exists(DEM_GER_path):
    pass
else:
    # create vrt for Germany with all downloaded tiles
    vrt = gdal.BuildVRT(path_safe(DEM_GER_path), getFilelist(storPath_DEM_TILES, ',tif'), separate = False)
    vrt = None
    convertVRTpathsTOrelative(f'{storPath_DEM}vrt_and_derivates/DEM_GER_S2.vrt')


if os.path.exists(slope_path):
    pass
else:
    #### calculate slope and aspect
    dem = xdem.DEM(DEM_GER_path)
    slope = dem.slope()
    aspect = dem.aspect()
    opts = {
        "BIGTIFF": "YES",
        "COMPRESS": "DEFLATE", 
        "TILED": "YES"
    }


    slope.to_file(slope_path, co_opts=opts)
    aspect.to_file(aspect_path, co_opts=opts)

    #### create lat and lon
if os.path.exists(lat_path):
    pass
else:
    # get metadata
    with rasterio.open(DEM_GER_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs_src = src.crs 

    # create grid
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))

    # get center coordinates of pixel
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    xs = np.array(xs).reshape((height, width))
    ys = np.array(ys).reshape((height, width))

    # transform to wgs84
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(xs, ys)

    # export
    out_meta = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs_src,    
        "transform": transform,
        "nodata": -9999
    }

    with rasterio.open(lat_path, 'w', **out_meta) as dst:
        dst.write(lons.astype('float32'), 1)

    with rasterio.open(lon_path, 'w', **out_meta) as dst:
        dst.write(lats.astype('float32'), 1)


if len(getFilelist(f"{storPath_TILES}Slope", '.tif', deep=True)) != 0:
    pass
else:
    ##### cut SLOPE, ASPECT, LAT and LON into FORCE TILES (use DEM tiles as template)
    # set paths and search pattern for tile ending
    tiles = getFilelist(storPath_DEM_TILES, ',tif')
    pattern = re.compile(r'X\d{4}_Y\d{4}')

    # loop over tiles and cut tiles from germany-wide tifs
    for tile_path in tiles:
        tile_id = pattern.search(tile_path).group()
        
        # Get tile extent
        with rasterio.open(tile_path) as tile:
            bounds = tile.bounds
            tile_crs = tile.crs

        for suffix, large_file in zip(['Slope', 'Aspect', 'LAT', 'LON'], [slope_path, aspect_path, lat_path, lon_path]):
            
            tile_out_path = path_safe(f"{storPath_TILES}{suffix}/{suffix}_{tile_id}.tif")

            with rasterio.open(large_file) as src:
                window = from_bounds(*bounds, transform=src.transform)
                data   = src.read(window=window)
                transform = src.window_transform(window)

                profile = src.profile.copy()
                profile.update({
                    "height":    data.shape[1],
                    "width":     data.shape[2],
                    "transform": transform,
                })

                with rasterio.open(tile_out_path, "w", **profile) as dst:
                    dst.write(data)

    ##### warp DEM, SLOPE, ASPECT, LAT and LON to LST for air temp correction

    for dname, gpath in zip(['DEM', 'ASPECT', 'SLOPE', 'LAT', 'LON'], [DEM_GER_path, aspect_path, slope_path, lat_path, lon_path]):
        warp_raster_to_reference(gpath, path_to_S3_template, path_safe(f"{storPath_DEM}reprojected/{dname}_GER_S3.tif"))



################################################################### ERA5 2m AirTemp and Geopotential to S3 warp

# set variables and parameter
storPath_ERA5 = path_safe(f"{storPath_master}ERA5/")
path_to_geopot = path_safe(f"{storPath_ERA5}/tif/Geopot/S3_res/geopotential_S3.tif")
path_to_geopot_raw = f"{storPath_ERA5}grib/geopotential/geopotential_unique.grib"

# first, warp geopotential as it is needed for 2m AirTemp
if os.path.exists(f"{path_to_geopot}.tif"):
    pass
else:
    warp_ERA5_to_reference(path_to_geopot_raw, path_to_S3_template, path_to_geopot, bandL=[1])


# first, ERA5 2m air temperature needs to be warped to Sentinel-3 resolution
storPath = '/data/Aldhani/eoagritwin/et/Auxiliary/ERA5/tiff/'
base_path = '/data/Aldhani/eoagritwin/et/Auxiliary/ERA5/grib/'

dem_path = '/data/Aldhani/eoagritwin/et/Auxiliary/DEM/reprojected/DEM_GER_LST_WARP.tif'
 = '/data/Aldhani/eoagritwin/et/Auxiliary/ERA5/tiff/low_res_rss/geopotential/geopotential_low_res.tif'

for variable in ['2m_temperature']:#]:'geopotential', 
    if variable != 'geopotential': # geopotential is constant over time --> only a single layer needed
        for year in [2019]:#, 2022, 1):
            files = [file for file in getFilelist(os.path.join(base_path, variable), '.grib') if str(year) in file]
            for file in files:
                # make folder to store data and subset files only to growing season
                outDir = f"{storPath}/2m_temperature/{year}/"
                os.makedirs(outDir, exist_ok=True)
                
                m = int(file.split('_')[-1].split('.')[0])
                if growingSeasonChecker(m):
                    month = INT_TO_MONTH[f'{m:02d}']
                    outPath = f"{outDir}{variable}_{year}_{month}.tif"
                    if os.path.exists(outPath):
                        print(f"{variable} already processes for {month}/{year}")
                        continue
                    else:
                        warp_ERA5_to_reference(file, path_to_S3_template, outPath, sharp_DEM=dem_path,
                                               sharp_geopot=geopot_path,
                                               sharp_blendheight=100, sharp_rate=STANDARD_ADIABAT)
    
    else:
        




################################################################### Sentinel-3 compositing

# - '/home/potzschf/repos/evapo_scripts/Sentinel/py/make_masked_LST_stacks.py'



# # Sentinel-2 compositing and tiling



# ##### preprocessing that could be done once, if products could be stored:
# # reproject/resample/align DEM with Sentinel-2 & derive slope, aspect
# # calculate lat/lon
# # prepare thuenen maps 
# - '/home/potzschf/repos/evapo_scripts/Guzinski/aux/DEM_to_FORCE_and_OTHERS.ipynb' # also creates lat/lon
# - '/home/potzschf/repos/evapo_scripts/Guzinski/aux/ERA_5_2m_AIR_GEOPOT_for_LST_QA.py'