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

driver_mem_ras = gdal.GetDriverByName('MEM')  # In-memory raster
driver_gpkg = ogr.GetDriverByName('GPKG')
driver_mem_vec = ogr.GetDriverByName("Memory")
from FieldWaterUseTools.FuncBox.FieldFuncis import unique_dict, make2000000000
from FieldWaterUseTools.FuncBox.Misc import assert_same_length, getFilelist, path_safe, makeTif_np_to_matching_tif, getSpatRefRas


# set paths and parameter
year = 2023
states = ['Brandenburg']#, 'Brandenburg']
models = ['FromScratch_dilate_T']#, 'FromScratch_dilate_T']
t_exts = ['07']#, '03']
t_bounds = ['01']#, '01']
maskVersions = ['ThuenenMasked']#, 'UnMasked']

assert_same_length(states, models, t_exts, t_bounds, maskVersions)

for idx, state in enumerate(states):
    # load GER shp and subset shp to state
    if state != 'Germany':
        ger = gpd.read_file(f'{origin}misc/gadm41_DEU_shp/gadm41_DEU_1.shp')
        aoi = ger[ger['NAME_1'] == state]
    else:
        aoi = gpd.read_file(f'{origin}misc/gadm41_DEU_shp/gadm41_DEU_0.shp')
        
    seg_path = f"{origin}fields/06_Segmentation/{state}/{models[idx]}/{year}/{maskVersions[idx]}/"

    # get tile files for parameter combination
    para_id = f"ext_{t_exts[idx]}_bound_{t_bounds[idx]}"
    files_sub = [file for file in getFilelist(f"{seg_path}Tiles/", '.tif') if para_id in file]

    # ensure that aoi of state has same crs as segmented tifs
    vrt_ds = gdal.Open(f"{seg_path}vrt/{maskVersions[idx]}_{para_id}.vrt")
    gt = vrt_ds.GetGeoTransform()
    proj = vrt_ds.GetProjection()
    xsize = vrt_ds.RasterXSize
    ysize = vrt_ds.RasterYSize
    # vrt_arr = vrt_ds.GetRasterBand(1).ReadAsArray()

    # reproject aoi and convert gdf to OGR layer in memory for easier handling
    aoi = aoi.to_crs(proj)
    mem_ds = driver_mem_vec.CreateDataSource("memAoi")
    mem_layer = mem_ds.CreateLayer("aoi", srs=ogr.osr.SpatialReference(wkt=proj))
    mem_layer_defn = mem_layer.GetLayerDefn()
    for geom in aoi.geometry:
        feat = ogr.Feature(mem_layer_defn)
        feat.SetGeometry(ogr.CreateGeometryFromWkb(geom.wkb))
        mem_layer.CreateFeature(feat)
        feat = None

    # create binary mask from aoi
    mem_raster = driver_mem_ras.Create("", xsize, ysize, 1, gdal.GDT_Byte)
    mem_raster.SetGeoTransform(gt)
    mem_raster.SetProjection(proj)
    band = mem_raster.GetRasterBand(1)
    band.Fill(0)
    band.SetNoDataValue(0)
    gdal.RasterizeLayer(mem_raster, [1], mem_layer, burn_values=[1])
    aoi_mask_arr = mem_raster.ReadAsArray().astype(np.uint8)
    # npTOdisk(vrt_arr * mask_arr, export_path, f'{path}prediction/{path.split('/')[-2]}.tif', bands=1, noData=0)


    # make a physical copy of files_sub, in case the relabelling goes wrong, as well as change to int64 as in large states or germany, the dirty offset needs to be very high
    outPath = f"{origin}fields/07_Polygonized/{state}/{models[idx]}/{year}/{maskVersions[idx]}/"
    tempPath = path_safe(f"{outPath}temp/")
    if len(getFilelist(tempPath, '.tif')) != 0:
        for file in getFilelist(tempPath, '.tif'):
            os.remove(file)
    for file in files_sub:
        dest = f"{tempPath}{file.split('/')[-1]}"
        with rasterio.open(file) as src:
            profile = src.profile.copy()
            profile.update(dtype=rasterio.int64)
            with rasterio.open(dest, 'w', **profile) as dst:
                dst.write(src.read().astype(np.int64))

    # loop over files to get row and col cuts
    rows, cols = [], []
    for file in files_sub:
        rows.append(int(file.split(para_id)[-1].split('_')[1]))
        cols.append(int(file.split(para_id)[-1].split('_')[2].split('.tif')[0]))
    rows.sort()
    cols.sort()
    rows = list(set(rows))
    cols = list(set(cols))
    rows.sort()
    cols.sort()

    ##### iterate over possible row/col connections and search for fields in neighbouring tiles
    # start at top-left-corner

    d_offset = 2000000000# needed to identify poylgons that reach over tile borders

    for row in rows:
        for col in cols:

            tile1 = [sub for sub in files_sub if f'_{row}_{col}.tif' in sub]
            if len(tile1) == 0:
                continue
            else: 
                ### check for neighbouring tiles and store result in list
                #print(f'_{row}_{col}.tif')
                # upper  (row -1)
                if rows.index(row)!=0:
                    upper = [sub for sub in files_sub if f'_{rows[rows.index(row)-1]}_{col}.tif' in sub]
                else:
                    upper = []

                # upper right (row -1 and col +1)
                if rows.index(row)!=0 and cols.index(col) < len(cols) - 1:
                    upper_right = [sub for sub in files_sub if f'_{rows[rows.index(row)-1]}_{cols[cols.index(col)+1]}.tif' in sub]
                else:
                    upper_right = []

                # right (col +1)
                if cols.index(col) < len(cols) - 1:
                    right = [sub for sub in files_sub if f'_{row}_{cols[cols.index(col)+1]}.tif' in sub]
                else:
                    right = []

                # lower right (row +1 and col +1)
                if rows.index(row) < len(rows) - 1 and cols.index(col) < len(cols) - 1:
                    lower_right = [sub for sub in files_sub if f'_{rows[rows.index(row)+1]}_{cols[cols.index(col)+1]}.tif' in sub]
                else:
                    lower_right = []

                # bottom (row +1)
                if rows.index(row) < len(rows) - 1:
                    bottom = [sub for sub in files_sub if f'_{rows[rows.index(row)+1]}_{col}.tif' in sub]
                else:
                    bottom = []

                if any(len(lst) > 0 for lst in [upper, upper_right, right, lower_right, bottom]):
                    # load starting tile
                    ds = gdal.Open(tile1[0])
                    dat = ds.GetRasterBand(1).ReadAsArray()
                    
                    ### check the other tiles
                    # upper
                    if len(upper) == 1:
                        ds = gdal.Open(upper[0])
                        neighbour = ds.GetRasterBand(1).ReadAsArray()
                        
                        unique_pairs = np.unique(np.stack((dat[0,:], neighbour[-1,:]), axis=1), axis=0)
                        valid_pairs = unique_pairs[(unique_pairs != 0).all(axis=1)]
                        valid_dict = unique_dict(valid_pairs)
                        for v, p_list in valid_dict.items():
                            dat[dat == v] = d_offset + v
                            for p in p_list:
                                neighbour[neighbour == p] = d_offset + v
                        # export manipulated tile (will overwrite)
                        makeTif_np_to_matching_tif(neighbour, upper[0], upper[0], noData=0)

                    # upper right
                    if len(upper_right) == 1:
                        ds = gdal.Open(upper_right[0])
                        neighbour = ds.GetRasterBand(1).ReadAsArray()
                        
                        unique_pairs = np.unique(np.stack((dat[0,-1], neighbour[-1,0])), axis=0)
                        valid_pairs = unique_pairs[(unique_pairs != 0).all()]
                        if len(valid_pairs) > 1:
                            for v, p in valid_pairs:
                                dat[dat == v] = d_offset + v
                                neighbour[neighbour == p] = d_offset + v
                        # export manipulated tile (will overwrite)
                        makeTif_np_to_matching_tif(neighbour, upper_right[0], upper_right[0], noData=0)

                    # right
                    if len(right) == 1:
                        ds = gdal.Open(right[0])
                        neighbour = ds.GetRasterBand(1).ReadAsArray()
                        
                        unique_pairs = np.unique(np.stack((dat[:,-1], neighbour[:,0]), axis=1), axis=0)
                        valid_pairs = unique_pairs[(unique_pairs != 0).all(axis=1)]
                        valid_dict = unique_dict(valid_pairs)
                        for v, p_list in valid_dict.items():
                            dat[dat == v] = d_offset + v
                            for p in p_list:
                                neighbour[neighbour == p] = d_offset + v
                        # export manipulated tile (will overwrite)
                        makeTif_np_to_matching_tif(neighbour, right[0], right[0], noData=0)

                    # lower right
                    if len(lower_right) == 1:
                        ds = gdal.Open(lower_right[0])
                        neighbour = ds.GetRasterBand(1).ReadAsArray()
                        
                        unique_pairs = np.unique(np.stack((dat[-1,-1], neighbour[0,0])), axis=0)
                        valid_pairs = unique_pairs[(unique_pairs != 0).all()]
                        if len(valid_pairs) > 1:
                            for v, p in valid_pairs:
                                dat[dat == v] = d_offset + v
                                neighbour[neighbour == p] = d_offset + v
                        # export manipulated tile (will overwrite)
                        makeTif_np_to_matching_tif(neighbour, lower_right[0], lower_right[0], noData=0)

                    # bottom
                    if len(bottom) == 1:
                        ds = gdal.Open(bottom[0])
                        neighbour = ds.GetRasterBand(1).ReadAsArray()

                        unique_pairs = np.unique(np.stack((dat[-1,:], neighbour[0,:]), axis=1), axis=0)
                        valid_pairs = unique_pairs[(unique_pairs != 0).all(axis=1)]
                        valid_dict = unique_dict(valid_pairs)
                        for v, p_list in valid_dict.items():
                            dat[dat == v] = d_offset + v
                            for p in p_list:
                                neighbour[neighbour == p] = d_offset + v
                        # export manipulated tile (will overwrite)
                        makeTif_np_to_matching_tif(neighbour, bottom[0], bottom[0], noData=0)

                    makeTif_np_to_matching_tif(dat, tile1[0], tile1[0], noData=0)
                else:
                    continue

    # loop over every tile again and clean up mess (e.g. 20000059, 40000059, 60000059)
    vec_func = np.vectorize(make2000000000, otypes=[int])
    for file in files_sub:
        ds = gdal.Open(file)
        arr = ds.GetRasterBand(1).ReadAsArray()
        mask = arr > d_offset * 2
        arr[mask] = vec_func(arr[mask])
        makeTif_np_to_matching_tif(arr, file, file, noData=0)

    # make a vrt and redo rasterIDs
    gdal.BuildVRT(f'{tempPath}quick_n_dirty.vrt', files_sub)
    vrt = None

    ds = gdal.Open(f'{tempPath}quick_n_dirty.vrt')
    block = ds.GetRasterBand(1).ReadAsArray()
    relabelled = measure.label(block, background=0, connectivity=1)
    print('label mess cleaned up')

    # create an in-memory raster-band with geoinfo of the relabelled array
    rows, cols = relabelled.shape
    relabelled_masked = relabelled * aoi_mask_arr # think here about on how to avoid cutting of fields at the edges!!!!
    del relabelled, block

    raster_ds = driver_mem_ras.Create('', cols, rows, 1, gdal.GDT_Int32)
    raster_ds.GetRasterBand(1).WriteArray(relabelled_masked)
    raster_ds.SetGeoTransform(ds.GetGeoTransform())
    raster_ds.SetProjection(ds.GetProjection())
    src_band = raster_ds.GetRasterBand(1)

    # create a mask for the background (otherwise it would get a polygon as well)
    mask_array = (relabelled_masked != 0).astype(np.uint8)
    mask_ds = driver_mem_ras.Create('', cols, rows, 1, gdal.GDT_Int32)
    mask_ds.GetRasterBand(1).WriteArray(mask_array)
    mask_band = mask_ds.GetRasterBand(1)
    

    # create output 
    out_ds = driver_gpkg.CreateDataSource(f'{outPath}{maskVersions[idx]}_{para_id}_10m.gpkg')  # Output vector file
    out_layer = out_ds.CreateLayer('polygons', getSpatRefRas(ds), geom_type=ogr.wkbPolygon)
    field_defn = ogr.FieldDefn('FieldID', ogr.OFTInteger)
    out_layer.CreateField(field_defn)

    # polygonize
    gdal.Polygonize(src_band, mask_band, out_layer, 0, [], callback=None)
    del out_ds
    print('fields poylgonized')

    # store a masked version of rasterized and relabelled IACS 
    out_ds = gdal.GetDriverByName('GTiff').Create(path_safe(f"{seg_path}relabelled/{state}_{models[idx]}_{year}_{maskVersions[idx]}_{para_id}.tif"), cols, rows, 1, gdal.GDT_Int32)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    out_ds.GetRasterBand(1).WriteArray(relabelled_masked * mask_array)
    del out_ds
  
        
    # # donwsample
    # gdal.Warp("/vsimem/resampled.tif", raster_ds, xRes=2.5, yRes=2.5, resampleAlg='lanczos')
    # resampled_ds = gdal.Open("/vsimem/resampled.tif")
    # src_band = resampled_ds.GetRasterBand(1)
    # resampled_relabelled_masked = src_band.ReadAsArray()
    # rows, cols = resampled_relabelled_masked.shape
    # # create a mask for the background (otherwise it would get a polygon as well)
    # mask_array = (resampled_relabelled_masked != 0).astype(np.uint8)
    # mask_ds = driver_mem_ras.Create('', cols, rows, 1, gdal.GDT_Int32)
    # mask_ds.GetRasterBand(1).WriteArray(mask_array)
    # mask_band = mask_ds.GetRasterBand(1)
    # del mask_array

    # # create output 
    # out_ds = driver_gpkg.CreateDataSource(f'{outPath}{maskVersions[idx]}_{para_id}_resampled_2_5m.gpkg')  # Output vector file
    # out_layer = out_ds.CreateLayer('polygons', getSpatRefRas(ds), geom_type=ogr.wkbPolygon)
    # field_defn = ogr.FieldDefn('FieldID', ogr.OFTInteger)
    # out_layer.CreateField(field_defn)

    # # polygonize
    # gdal.Polygonize(src_band, mask_band, out_layer, 0, [], callback=None)
    # del out_ds
    # print('fields poylgonized')


    # clean up temp folder
    for file in getFilelist(tempPath, '.vrt'):
        os.remove(file)
    for file in getFilelist(tempPath, '.tif'):
        os.remove(file)
    for file in getFilelist(tempPath, '.xml'):
        os.remove(file)