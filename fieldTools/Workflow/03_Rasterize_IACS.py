import sys
import os
origin = '/workspace/'
sys.path.append('/media/')


from FieldWaterUseTools.FuncBox.Misc import getFilelist, path_safe
from FieldWaterUseTools.FuncBox.Polygons_to_Labels import *
from FieldWaterUseTools.FuncBox.DICT_LIST import EXCLUDE_LIST
from FieldWaterUseTools.FuncBox.ForceFuncis import force_order_Colors_for_VRT, force_to_vrt, reduce_forceTSA_output_to_validmonths

# set the year and state to create IACS masks and boundaries

fed_state = 'Brandenburg'

state_lkup = {'Brandenburg': ['BRB', '.geoparquet', 'EC_hcat_n'],
              'Niedersachsen': ['LSA', '.geoparquet','EC_hcat_n'],
              'MV': ['MV', '.shp','NU_BEZ'],
              'NRW': ['NRW','.geoparquet','EC_hcat_n'],
              'Saarland': ['SL', '.shp', 'BEZ']}


year = 2023

state = state_lkup[fed_state][0]
state_year = f'{state}_{year}'


polygon_path = [file for file in getFilelist(f'/data/{origin}fields/IACS/1_Polygons/{state_lkup[fed_state][0]}/',\
                                             state_lkup[fed_state][1]) if str(year) in file][0]

vrt_out = path_safe(f'/data/{origin}fields/Auxiliary/vrt/{state}/{year}/')
reduced_files = reduce_forceTSA_output_to_validmonths(f'/data/{origin}force/output/{state}/{year}/', 3, 8)
ordered_files = force_order_Colors_for_VRT(reduced_files, ['BLU', 'GRN', 'RED', 'BNR'], [f'MONTH-{d:02d}' for d in range(3,9,1)])


if os.path.isdir(vrt_out):
    if len(getFilelist(f'{vrt_out}', '.vrt', deep=True)) > 0:
        print('VRT seems to be already computated, probably to create masks based on IACS')
    else:
        force_to_vrt(reduced_files, ordered_files, vrt_out, True, bandnames=['BLU', 'GRN', 'RED', 'BNR'])
else:
    os.makedirs(vrt_out)
    force_to_vrt(reduced_files, ordered_files, vrt_out, True, bandnames=['BLU', 'GRN', 'RED', 'BNR'])


vrt_path = getFilelist(f"/data/{origin}fields/Auxiliary/vrt/{state}/{year}/{dirfinder(f'/data/{origin}/fields/Auxiliary/vrt/{state}/{year}/')[0]}",
                       '.vrt', deep=True)[0]


lines_out_gpkg_path = f'/data/{origin}fields/IACS/2_Lines/{state}/{year}/IACS_{state_year}.gpkg'
raster_lines_out_path = f'/data/{origin}fields/IACS/3_Rasterized_lines/{state}/{year}/IACS_{state_year}'
crop_mask_out_path = f'/data/{origin}fields/IACS/4_Crop_mask/{state}/{year}/IACS_{state_year}_cropMask'
# convert lines to polyongs
polygons_to_lines(polygon_path,
                  path_safe(lines_out_gpkg_path),
                  categories=EXCLUDE_LIST,
                  category_col=state_lkup[fed_state][2])

# rasterize lines
rasterize_lines(lines_out_gpkg_path, 
                vrt_path, 
                path_safe(f'{raster_lines_out_path}_lines_touch_true.tif'), all_touch=True)

rasterize_lines(lines_out_gpkg_path,
                vrt_path, 
                path_safe(f'{raster_lines_out_path}_lines_touch_false.tif'), all_touch=False)

# make a crop mask
make_crop_mask(path_to_polygon=polygon_path, 
               path_to_rasterized_lines=f'{raster_lines_out_path}_lines_touch_true.tif', 
               path_to_extent_raster=vrt_path, 
               path_to_mask_out=path_safe(f'{crop_mask_out_path}_lines_touch_true_crop_touch_true.tif'),
               all_touch=True,
               categories=EXCLUDE_LIST,
               category_col=state_lkup[fed_state][2])

make_crop_mask(path_to_polygon=polygon_path, 
               path_to_rasterized_lines=f'{raster_lines_out_path}_lines_touch_false.tif', 
               path_to_extent_raster=vrt_path, 
               path_to_mask_out=path_safe(f'{crop_mask_out_path}_lines_touch_false_crop_touch_true.tif'),
               all_touch=True,
               categories=EXCLUDE_LIST,
               category_col=state_lkup[fed_state][2])




make_crop_mask(path_to_polygon=polygon_path, 
               path_to_rasterized_lines= f'{raster_lines_out_path}_lines_touch_true.tif', 
               path_to_extent_raster=vrt_path, 
               path_to_mask_out=path_safe(f'{crop_mask_out_path}_cropMask_lines_touch_true_crop_touch_false.tif'),
               all_touch=False,
               categories=EXCLUDE_LIST,
               category_col=state_lkup[fed_state][2])

make_crop_mask(path_to_polygon=polygon_path, 
               path_to_rasterized_lines= f'{raster_lines_out_path}_lines_touch_false.tif', 
               path_to_extent_raster=vrt_path, 
               path_to_mask_out=path_safe(f'{crop_mask_out_path}_cropMask_lines_touch_false_crop_touch_false.tif'),
               all_touch=False,
               categories=EXCLUDE_LIST,
               category_col=state_lkup[fed_state][2])



# # This should happen at tile level, when we use S2 to finetune model --> labels needed for training
# # rasterized lines to 3D label
# print('take the long way')
# make_multitask_labels(f'/data/{origin}fields/IACS/3_Rasterized_lines/GSA-DE_{spat_temp_id}_lines_touch_true.tif', 
#                       f'/data/{origin}fields/IACS/5_Multitask_labels/GSA-DE_{spat_temp_id}_lines_touch_true_MTSK.tif')

# make_multitask_labels(f'/data/{origin}fields/IACS/3_Rasterized_lines/GSA-DE_{spat_temp_id}_lines_touch_false.tif', 
#                        f'/data/{origin}fields/IACS/5_Multitask_labels/GSA-DE_{spat_temp_id}_lines_touch_false_MTSK.tif')