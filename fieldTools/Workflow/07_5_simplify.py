import os
import shutil
from skimage import measure
import numpy as np
import geopandas as gpd
from osgeo import gdal, ogr
import rasterio
import sys

origin = '/workspace/'
sys.path.append('/media/')

from FieldWaterUseTools.FuncBox.Misc import assert_same_length, getFilelist, path_safe, makeTif_np_to_matching_tif, getSpatRefRas, getSpatRefVec

year = 2023
states = ['Brandenburg', 'Brandenburg']
models = ['FromScratch_dilate_T', 'FromScratch_dilate_T']
t_exts = ['03', '03']
t_bounds = ['01', '01']
maskVersions = ['ThuenenMasked', 'UnMasked']

idx = 0
state = states[idx]
para_id = f"ext_{t_exts[idx]}_bound_{t_bounds[idx]}"
outPath = f"{origin}fields/07_Polygonized/{state}/{models[idx]}/{year}/{maskVersions[idx]}/"
f"{outPath}{maskVersions[idx]}_{para_id}.gpkg"


# load vector ds to be simplified
vector_path = f"{outPath}{maskVersions[idx]}_{para_id}.gpkg"
in_ds = ogr.Open(vector_path)
layer = in_ds.GetLayer(0)  # or "polygons"?

# create output layer
outPath = f"{vector_path.split('.')[0]}_True_simple10_no_buff.gpkg"
driver = ogr.GetDriverByName('GPKG')  # or 'GeoJSON', 'GPKG', etc.
out_ds = driver.CreateDataSource(outPath)  # Output vector file
out_layer = out_ds.CreateLayer('polygons', getSpatRefVec(in_ds), geom_type=ogr.wkbPolygon)

# copy layer fields
layer_defn = layer.GetLayerDefn()
for i in range(layer_defn.GetFieldCount()):
    field_defn = layer_defn.GetFieldDefn(i)
    out_layer.CreateField(field_defn)



for idx1, feature in enumerate(layer):

    if idx1 % 1000 == 0:
        print(idx1)
    geom = feature.GetGeometryRef()

    if geom is None:  # skip null geometries
        continue
    simplified = geom.Simplify(10)#geom.Buffer(2).Buffer(-2).SimplifyPreserveTopology(11)
    if simplified is None or simplified.IsEmpty():  # skip if simplify collapsed it
        continue
    
    new_feature = ogr.Feature(out_layer.GetLayerDefn())
    new_feature.SetGeometry(simplified)
    # copy attributes if needed
    for i in range(feature.GetFieldCount()):
        val = feature.GetField(i)
        if val is not None:
            new_feature.SetField(i, val)
    out_layer.CreateFeature(new_feature)

out_layer = None
out_ds = None  # forces flush to disk