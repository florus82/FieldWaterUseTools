### create a 1500 x 1500 pixel reference vector dataset that covers Germany (EPSG: 3035). 
# Grid is based on FORCE Germany grid system for potential future compability
### the reference grid will be calculated from the extents of DEM tiles that were cutted into FORCE grids by a FORCE function 
# (the grid vector file on
### Aldhani contains a rounding error that produces sometimes 1501x1500 tiles)

from Misc import getFilelist
import rasterio
import geopandas as gpd
import os
import sys
from shapely.geometry import box

origin = '/workspace/'
sys.path.append('/media/')

path_to_tiles = f"{origin}et/Auxiliary/DEM/Force_Tiles/DEM/"
tiles = getFilelist(path_to_tiles, ".tif")
conti = []

for tile in tiles:
    with rasterio.open(tile) as src:
        bounds = src.bounds  # left, bottom, right, top
        geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        crs = src.crs

        conti.append({
            "filename": tile.split("DEM_")[-1].split(".tif")[0],
            "geometry": geom,
            "crs_used": str(crs)
        })

gdf = gpd.GeoDataFrame(conti, geometry="geometry", crs=conti[0]["crs_used"])
gdf.to_file(f"{origin}et/Auxiliary/GRID/Gridding_ET_FORCE.gpkg", driver="GPKG", layer="tiles")