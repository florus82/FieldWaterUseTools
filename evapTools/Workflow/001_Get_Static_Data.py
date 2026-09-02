import os
import sys
import time
import requests
import openeo
import math
from rasterio.transform import from_origin
from osgeo import gdal
import rasterio 
import geopandas as gpd
import numpy as np

# Static dataset needed are a DEM for Germany and derivates

origin = '/data/Aldhani/eoagritwin/et/'
sys.path.append('/home/potzschf/repos/')


from FieldWaterUseTools.FuncBox.Misc import getFilelist, path_safe, slash_checker # 
from FieldWaterUseTools.FuncBox.DICT_LIST import REAL_INT_TO_MONTH # 


# set masterpath for stored data
storPath_master = slash_checker(path_safe(f"{origin}Z_REPO_TEST/"))#'/place/to/store/porducts/')
path_to_grid = f"{storPath_master}GRID/Gridding_ET_FORCE.gpkg"

######################################################################### DEM

storPath_DEM = slash_checker(path_safe(f"{storPath_master}DEM/raw_tiles/"))
storPath_DEM_tiled = slash_checker(path_safe(f"{storPath_master}DEM/Gridded/"))
connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()

overlap = 0.01 # otherwise there might be small gaps between tiles

# while len(getFilelist(storPath_DEM, '.tif')) < 119:
#     for i in range(0,12,1):
#         for j in range(0,10,1):
#             if i == 9 and j == 0:
#                  continue
#             aoi = {'west': 5 + i - overlap, 'south': 46 + j - overlap, 'east': 6 + i + overlap, 'north': 47 + j + overlap}
#             storP = f"{storPath_DEM}DEM_GER_{j}_{i}.tif"

#             if os.path.exists(storP):
#                     t = time.localtime()
#                     ti = time.strftime("%H:%M:%S", t)
#                     print(f"already exists - next one at {ti}")

#             else:
#                 try:
#                     dem_cube = connection.load_collection(
#                                     "COPERNICUS_30",
#                                     spatial_extent = aoi,
#                                     bands=["DEM"]
#                                     )                               
#                     dem_cube.download(storP)

#                 except Exception as e:
#                     print(e)
#                     t = time.localtime()
#                     ti = time.strftime("%H:%M:%S", t)
#                     print(f"thrown at {ti}")
#                     continue

##### create lat/lon from grid
storPath_LAT = slash_checker(path_safe(f"{storPath_master}LAT/GER/"))
storPath_LAT_tiled = slash_checker(path_safe(f"{storPath_master}LAT/Gridded/"))
storPath_LON = slash_checker(path_safe(f"{storPath_master}LON/GER/"))
storPath_LON_tiled = slash_checker(path_safe(f"{storPath_master}LON/Gridded/"))

# get metadata
grid_GER = gpd.read_file(path_to_grid)

minx, miny, maxx, maxy = grid_GER.total_bounds

pixel_size = 20 # as gpd was created from raster at this resolution

n_pixels_x = (maxx - minx) / pixel_size
n_pixels_y = (maxy - miny) / pixel_size

width = int(math.ceil(n_pixels_x))
height = int(math.ceil(n_pixels_y))

# Raster transform
transform = from_origin(
    minx,
    maxy,
    pixel_size,
    pixel_size
)

# create grid
cols, rows = np.meshgrid(np.arange(width), np.arange(height))
# create grid
cols, rows = np.meshgrid(np.arange(width), np.arange(height))

print('created meshs')
# get center coordinates of pixel
xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
xs = np.array(xs).reshape((height, width))
ys = np.array(ys).reshape((height, width))

print('transformed')
# export
out_meta = {
    "driver": "GTiff",
    "height": height,
    "width": width,
    "count": 1,
    "dtype": "float32",
    "crs": grid_GER.crs,    
    "transform": transform,
    "nodata": -9999
}
with rasterio.open(f"{storPath_LON}LON_GER_4326.tif", 'w', **out_meta) as dst:
    dst.write(xs.astype('float32'), 1)

# Example to save lat raster
with rasterio.open(f"{storPath_LAT}LAT_GER_4326.tif", 'w', **out_meta) as dst:
    dst.write(ys.astype('float32'), 1)





vrt_name = f'{storPath_DEM}DEM.vrt'
vrt = gdal.BuildVRT(vrt_name, getFilelist(storPath_DEM, ".tif"), separate = False)
vrt = None

# # now cut DEM, aspect, slope into FORCE TILES
# grid_GER = gpd.read_file(path_to_grid)
# DEM_vrt = vrt_name
# with rasterio.open(DEM_vrt) as src:

#     for tileID in grid_GER['filename']:
#         tile = grid_GER[grid_GER['filename'] == tileID]
#         # print(Vector.crs)
#         out_image, out_transform=mask(src,tile.geometry,crop=True)
#         out_meta=src.meta.copy() # copy the metadata of the source DEM
        
#         out_meta.update({
#             "driver":"Gtiff",
#             "height":out_image.shape[1], # height starts with shape[1]
#             "width":out_image.shape[2], # width starts with shape[2]
#             "transform":out_transform
#         })
                
#         with rasterio.open(f'{storPath_DEM_tiled}/DEM_{tileID}.tif','w',**out_meta) as dst:
#             dst.write(out_image)