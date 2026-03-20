import sys

origin = '/workspace/'
sys.path.append('/media/')

import shutil
from skimage import measure
import geopandas as gpd
from FieldWaterUseTools.FuncBox.FieldFuncis import unique_dict, make2000000q


# load shapefiles and 
state = 'Brandenburg'
model = 'AI4_RGB_exclude_True_38'

year = 2019
ger = gpd.read_file(f'/data/{origin}misc/gadm41_DEU_shp/gadm41_DEU_0.shp')

if state != 'Germany':
    aoi = ger[ger['NAME_1'] == state]



paths = ['/data/Aldhani/eoagritwin/fields/segmented/Brandenburg/AI4_RGB_exclude_True_38/2023/masked_lines_touch_true_crop_touch_false_linecrop_text0.1_tbound0.7/',
         '/data/Aldhani/eoagritwin/fields/segmented/Brandenburg/AI4_RGB_exclude_True_38/2023/unmasked_text0.1_tbound0.65/']

combis = ['0.1_0.7', '0.1_0.65']




ext = '04'
bound = '02'
