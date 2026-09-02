import os
import osgeo
import xml.etree.ElementTree as ET
import numpy as np
import random

from osgeo import gdal, osr, ogr
from datetime import datetime, timezone
import requests

def getFilelist(originpath, ftyp, deep = False, order = True):
    '''
    Return a list of files that match the specified type
    
    :param originpath: string of path to folder to be searched
    :param ftyp: string of file ending/type to be searched for
    :param deep: if set to True, search will be recursive within folder
    :param order: if True, returned list will be sorted
    '''
    out   = []
    if deep == False:
        files = os.listdir(originpath)
        for i in files:
            if i.split('.')[-1] in ftyp:
                if originpath.endswith('/'):
                    out.append(originpath + i)
                else:
                    out.append(originpath + '/' + i)
            # else:
            #     print("non-matching file - {} - found".format(i.split('.')[-1]))
    else:
        for path, subdirs, files in os.walk(originpath):
            for i in files:
                if i.split('.')[-1] in ftyp:
                    out.append(os.path.join(path, i))
    if order == True:
        out = sorted(out)
    return out


def path_safe(path):
    """when storing a file, this function makes sure that the directory exists, where file will be stored

    Args:
        path (str): path to file (or directory)
    """
    if os.path.splitext(path)[1]: # checks if path points to a file
            dir_path = os.path.dirname(path)
    else:
        dir_path = path  # treat as directory

    if dir_path == "":
        print('this is not a path!!!')
        return path

    os.makedirs(dir_path, exist_ok=True)
    return path

def slash_checker(path):
    if not path.endswith('/'):
        return f"{path}/"
    else:
        return path

def dirfinder(path):
    """ returns a list with all directory names within a folder

    Args:
        path (str): str path to folder that will be searched for directories

    Returns:
        list: list of directory names (str)
    """
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def sortListwithOtherlist(list1, list2, rev=False):
    ''' list1: unsorted list
        list2: unsorted list with same length as list1
        Sorts list2 based on sorted(list1). Returns sorted list1 list2
        if rev == True, list will be returned reversed
        '''
    sortlist1, sortlist2 = zip(*sorted(zip(list1, list2)))
    sort1 = list(sortlist1)
    sort2 = list(sortlist2)

    if rev:
        sort1.reverse()
        sort2.reverse()
  
    return sort1, sort2


def get_row_col_indices(chipsize, overlap, number_of_rows, number_of_cols):
    '''
    chipsize: the desired size of image chips passed on to GPU for prediction
    overlap: the overlap in rows and cols of image chips @chipsize
    number_of_rows, number_of_cols: overall number of rows and cols of entire datablock that should be predicted
    '''
    row_start = [i for i in range(0, number_of_rows, chipsize - overlap)]
    row_end = [i for i in range (chipsize, number_of_rows, chipsize - overlap)]
    row_start = row_start[:len(row_end)] 

    col_start = [i for i in range(0, number_of_cols, chipsize - overlap)]
    col_end = [i for i in range (chipsize, number_of_cols, chipsize - overlap)] 
    col_start = col_start[:len(col_end)]

    return [row_start, row_end, col_start, col_end]

def download_thuenen_cropTypes(year, storpath, stacURL="https://eodata.thuenen.de/stac/api/v1/collections/crop-type-map-latest/items/crop-type-map-latest-"):
    '''
    year: year of crop type map (int)
    storpath: path to folder where crop type map should be stored (str)
    stacURL: if URL might change in the future please adapt
    '''
    
    stac_url = f"{stacURL}{year}"
    output_file = f"{slash_checker(storpath)}Thuenen_CropType_{year}.tif"

    # Get STAC metadata
    item = requests.get(stac_url)
    item.raise_for_status()
    item = item.json()


    # Pick the first TIFF asset
    tif_asset = next(
        asset for asset in item["assets"].values()
        if "tiff" in asset.get("type", "").lower()
    )

    url = tif_asset["href"]


    # Download entire COG
    with requests.get(url, stream=True) as r:
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0))
        downloaded = 0

        with open(output_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)


    print(f"Downloaded complete: {output_file}")


def getExtentRas(raster):
    if type(raster) is str:
        ds = gdal.Open(raster)
    elif type(raster) is gdal.Dataset:
        ds = raster
    gt = ds.GetGeoTransform()
    ext = {'Xmin': gt[0],
            'Xmax': gt[0] + (gt[1] * ds.RasterXSize),
            'Ymin': gt[3] + (gt[5] * ds.RasterYSize),
            'Ymax': gt[3]}
    return ext


def commonBoundsDim(extentList):
    # create empty dictionary with list slots for corner coordinates
    k = ['Xmin', 'Xmax', 'Ymin', 'Ymax']
    v = [[], [], [], []]
    res = dict(zip(k, v))

    # fill it with values of all raster files
    for i in extentList:
        for j in k:
            res[j].append(i[j])
    # determine min or max values per values' list to get common bounding box
    ff = [max, min, max, min]
    for i, j in enumerate(ff):
        res[k[i]] = j(res[k[i]])
    return res


def commonBoundsCoord(ext):
    if type(ext) is dict:
        ext = [ext]
    else:
        ext = ext
    cooL = []
    for i in ext:
        coo = {'UpperLeftXY': [i['Xmin'], i['Ymax']],
               'UpperRightXY': [i['Xmax'], i['Ymax']],
               'LowerRightXY': [i['Xmax'], i['Ymin']],
               'LowerLeftXY': [i['Xmin'], i['Ymin']]}
        cooL.append(coo)
    return cooL


def convertVRTpathsTOrelative(vrt_path):
    tree = ET.parse(vrt_path)
    root = tree.getroot()

    for source in root.findall(".//SourceFilename"):
        abs_path = source.text
        rel_path = os.path.relpath(abs_path, os.path.dirname(vrt_path))  # Convert to relative
        source.text = rel_path
        source.set("relativeToVRT", "1")  # Add the attribute

    # Save the modified VRT file
    tree.write(vrt_path)


def vrtPyramids(vrtpath):
    '''takes a vrtpath (or gdalOpened vrt) and produces pyramids'''
    if type(vrtpath) == osgeo.gdal.Dataset:
        image = vrtpath
    else:
        Image = gdal.Open(vrtpath, 0) # 0 = read-only, 1 = read-write. 
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
    Image.BuildOverviews("NEAREST", [2,4,8,16,32,64])
    del Image


def RasterKiller(raster_path):
    if os.path.isfile(raster_path):
        os.remove(raster_path)


def checkPath(path):
    if isinstance(path, str):
            return gdal.Open(path)
    else:
        return path


def stackReader(path_to_stack, bands=False, era=False):
    """Reads-in a raster stacks and returns a 3D numpy array of that array.
    Optionally, a list with band names will be returned

    Args:
        path_to_stack (str): path to the stack.tif
        bands (bool): If True, a list with band names of stack wil be returned as well. (WORKS ONLY WITH ERA5.grib files for now!!!!)
        era (bool): If bands True and era True, the bands metadata is extracted in a different manner
    """
    conti = []
    if type(path_to_stack) != osgeo.gdal.Dataset:
        ds = gdal.Open(path_to_stack)
    else:
        ds = path_to_stack
    ds = checkPath(path_to_stack)
    bandCount = ds.RasterCount
    if bands:
        bandsL = []
        if bandCount > 1:
            for b in range(bandCount):
                conti.append(ds.GetRasterBand(b+1).ReadAsArray())
                if not era:
                    bandsL.append(ds.GetRasterBand(b+1).GetDescription())
                else:
                    bandsL.append(datetime.fromtimestamp(int(ds.GetRasterBand(b+1).GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
            return np.dstack(conti), bandsL
        else:
            conti.append(ds.GetRasterBand(1).ReadAsArray())
            if not era:
                bandsL.append(ds.GetRasterBand(1).GetDescription())
            else:
                bandsL.append(datetime.fromtimestamp(int(ds.GetRasterBand(1).GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
            return np.dstack(conti), bandsL
    else:
        if bandCount > 1:
            for b in range(bandCount):
                conti.append(ds.GetRasterBand(b+1).ReadAsArray())
            return np.dstack(conti)
        else:
            return ds.GetRasterBand(1).ReadAsArray()
        

def stack_tifs(input_tif_list, output_tif=False, d_type=False):
    # Open the first raster to get geotransform, projection, and shape
    if type(input_tif_list) != osgeo.gdal.Dataset:
        src0 = gdal.Open(input_tif_list[0])
    else:
        src0 = input_tif_list
    x_size = src0.RasterXSize
    y_size = src0.RasterYSize
    proj = src0.GetProjection()
    geotrans = src0.GetGeoTransform()
    if d_type:
        dtype = d_type
    else:
        dtype = src0.GetRasterBand(1).DataType
    num_bands = len(input_tif_list)

    # Create output multi-band raster
    if output_tif:
        out_ds = gdal.GetDriverByName('GTiff').Create(output_tif, x_size, y_size, num_bands, dtype)
    else:
        out_ds = gdal.GetDriverByName('MEM').Create('', x_size, y_size, num_bands, dtype)
    out_ds.SetProjection(proj)
    out_ds.SetGeoTransform(geotrans)

    # Write each input raster as a band
    for i, tif_path in enumerate(input_tif_list):
        src = gdal.Open(tif_path)
        band_data = src.GetRasterBand(1).ReadAsArray()
        out_ds.GetRasterBand(i + 1).WriteArray(band_data)

    if output_tif:
        out_ds.FlushCache()
        out_ds = None  # Close file 
    else:
        return out_ds
    

def shuffle2Lists(list1, list2):
   
    paired = list(zip(list1, list2))
    random.shuffle(paired)
    list1_1, list2_2 = zip(*paired)

    list1 = list(list1_1)
    list2 = list(list2_2)

    return list1, list2 


def is_leap_year(year):
    return (year % 4 == 0) and (year % 100 != 0 or year % 400 == 0)


def warp_raster_to_reference(source_path, reference_path, output_path, resampling='bilinear', keepRes=False, dtype=None):
    '''
    source_path: the raster to be warped
    reference_path: the raster to which will be warped
    output_path: here the warped raster will be stored; if MEM is used, the warped raster will be returned as memory object
    resampling: method to do resampling, e.g. bilinear, cubic, nearest
    keepRes: if set to true the warp will be done without changing the resolution of the raster at source_path to that of reference_path; if set to an integer,
    the pixel size of reference_path will be divided by that integer to gain a new pixel size 
    dtype (gdal.GDT_): if None, the dtype from source_path will be used
    '''

    # Open reference raster
    ref_ds = checkPath(reference_path)
    ref_proj = ref_ds.GetProjection()
    ref_gt = ref_ds.GetGeoTransform()
    x_size = ref_ds.RasterXSize
    y_size = ref_ds.RasterYSize

    # Extract pixel size
    ref_x_res = ref_gt[1]
    ref_y_res = -ref_gt[5]  

    # Get bounds: xmin, ymin, xmax, ymax
    xmin = ref_gt[0]
    ymax = ref_gt[3]
    xmax = xmin + ref_x_res * x_size
    ymin = ymax - ref_y_res * y_size

    # If dtype not given, use dtype from source raster
    if dtype is None:
        src_ds = checkPath(source_path)
        dtype = src_ds.GetRasterBand(1).DataType  # matches gdal.GDT_* constants
        src_ds = None  # close

    if isinstance(keepRes, bool) and keepRes:
        src_ds = checkPath(source_path)
        src_gt = src_ds.GetGeoTransform()
        src_proj = osr.SpatialReference(wkt=src_ds.GetProjection())

        ref_proj = osr.SpatialReference(wkt=ref_ds.GetProjection())

        transform = osr.CoordinateTransformation(src_proj, ref_proj)

        # Pixel corners in source CRS
        px_width = src_gt[1]
        px_height = src_gt[5]  # usually negative

        # Point (0,0)
        x0, y0, _ = transform.TransformPoint(src_gt[0], src_gt[3])
        # Point (1 pixel right)
        x1, y1, _ = transform.TransformPoint(src_gt[0] + px_width, src_gt[3])
        # Point (1 pixel down)
        x2, y2, _ = transform.TransformPoint(src_gt[0], src_gt[3] + px_height)

        # Resolution in target CRS
        aim_x_res = ((x1 - x0)**2 + (y1 - y0)**2) ** 0.5
        aim_y_res = ((x2 - x0)**2 + (y2 - y0)**2) ** 0.5

    elif isinstance(keepRes, int) and not isinstance(keepRes, bool) and keepRes > 0:
        aim_x_res = ref_x_res / keepRes
        aim_y_res = ref_y_res / keepRes

    else:
        aim_x_res = ref_x_res
        aim_y_res = ref_y_res

    out_format = 'MEM' if output_path == 'MEM' else 'GTiff'
    # Set up warp options
    warp_options = gdal.WarpOptions(
        format=out_format,
        dstSRS=ref_proj,
        outputBounds=(xmin, ymin, xmax, ymax),
        xRes=aim_x_res,
        yRes=aim_y_res,
        resampleAlg=resampling,
        targetAlignedPixels=False,
        outputType=dtype
        # srcNodata=noDat,     
        # dstNodata=-999
    )


    # Perform reprojection and resampling
    warped_ds = gdal.Warp('', source_path, options=warp_options) if out_format == 'MEM' else gdal.Warp(output_path, source_path, options=warp_options)



    # gdal.Translate(
    #     output_path,
    #     temp_path,
    #     projWin=(xmin, ymax, xmax, ymin)
    # )
    # os.remove(temp_path)

    if output_path == 'MEM':
        return warped_ds
    else:
        pass #print(f"Raster warped and saved to: {output_path}")


def npTOdisk(arr, reference_path, outPath, bands = False, bandnames = False, noData = False, d_type = False):
    """exports a numpy array to a tif that is stored on disk

    Args:
        arr (numpy array): the array to be exported
        reference_path (str): path to the reference tif. The extent and dimensions must fit!!!!
        outPath (_str): path to exported tif on disk
    """
    ref_ds = checkPath(reference_path)
    ref_band = ref_ds.GetRasterBand(1)
    if not bands:
        bands = ref_ds.RasterCount
    if not d_type:
        out_ds = gdal.GetDriverByName('GTiff').Create(outPath, ref_ds.RasterXSize, ref_ds.RasterYSize, bands, ref_band.DataType)
    else:
        out_ds = gdal.GetDriverByName('GTiff').Create(outPath, ref_ds.RasterXSize, ref_ds.RasterYSize, bands, d_type)
    out_ds.SetGeoTransform(ref_ds.GetGeoTransform())
    out_ds.SetProjection(ref_ds.GetProjection())
    if bands == 1:
        out_ds.GetRasterBand(1).WriteArray(arr)
        if bandnames:
            out_ds.GetRasterBand(1).SetDescription(bandnames)
        if noData is not False:
            out_ds.GetRasterBand(1).SetNoDataValue(noData)
    else:
        for i in range(bands):
            out_ds.GetRasterBand(i+1).WriteArray(arr[:,:,i])
            if bandnames:
                out_ds.GetRasterBand(i+1).SetDescription(str(bandnames[i]))
            if noData is not False:
                out_ds.GetRasterBand(i+1).SetNoDataValue(noData)
    out_ds.FlushCache()


def getBandNames(rasterstack):
    bands = []
    ds = gdal.Open(rasterstack)
    numberBands = ds.RasterCount
    for i in range(numberBands):
        bands.append(ds.GetRasterBand(i+1).GetDescription())
    return bands


def makeTif_np_to_matching_tif(array, tif_path, path_to_file_out, noData = None, gdalType = None, bands=1):
    '''
    exports an np.array to a tif, based on a tif that has the same extent. Probably, the np.array is a manipulation of that tif
    array: the numpy array
    tif_path: path to the tif from which geoinformation will be extracted
    path_to_file_out: where the new tif should be stored
    noData = a no data value can be assigned to the exported tif
    gdaType = a different data type can be set here, otherwise, the one from hte tif at tif_path will be used
    bands = default single band raster, provide the integer of bands to export as stack
    '''
    ds = gdal.Open(tif_path)
    gtiff_driver = gdal.GetDriverByName('GTiff')
    no_data = ds.GetRasterBand(1).GetNoDataValue()
    if gdalType == None:
        dtypi = ds.GetRasterBand(1).DataType
    else:
        dtypi = gdalType

    out_ds = gtiff_driver.Create(path_to_file_out, ds.RasterXSize, ds.RasterYSize, bands, dtypi)
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())
    if bands == 1:
        out_ds.GetRasterBand(1).WriteArray(array)
        if noData != None:
            out_ds.GetRasterBand(1).SetNoDataValue(noData)
        else:
            if no_data is not None:
                out_ds.GetRasterBand(1).SetNoDataValue(no_data)
    else:
        for b in range(bands):
            out_ds.GetRasterBand(b + 1).WriteArray(array[:,:,b])
        if noData != None:
            for b in range(bands):
                out_ds.GetRasterBand(b + 1).SetNoDataValue(noData)
        else:
            if no_data is not None:
                out_ds.GetRasterBand(b + 1).SetNoDataValue(no_data)

    del out_ds


def maskVRT(vrtPath, maskArray, suffix):
    """Opens a vrt and masks it with a binary array of the same dimensions

    Args:
        vrtPath (str): path to the vrt
        maskArray (np.array): binary np array, where 1 == valid and 0 == maks
    """
    ds = gdal.Open(vrtPath)
    b = []
    for band in range(ds.RasterCount):
        b.append(ds.GetRasterBand(band + 1).ReadAsArray() * maskArray)
    masked_arr =  np.dstack(b)
    makeTif_np_to_matching_tif(masked_arr, vrtPath, f"{vrtPath.split('.')[0]}{suffix}.tif", bands=len(b))

def maskVRT_water(vrtPath, colorlist):
    """OLD!!!!Opens a vrt and applies dirty water mask --> slope = NA and aspect = 180
       NOW: a 5% threshold is applied on BNIR band

    Args:
        vrtPath (str): path to the vrt
        colorlist: the order in which the s2 bands are delivered. the bands of cube must start with them
    """
    ds = gdal.Open(vrtPath)
    b = []
    for band in range(ds.RasterCount):
        b.append(ds.GetRasterBand(band + 1).ReadAsArray())
    arr =  np.dstack(b)
    # mask = np.logical_and(arr[:,:,10] < 0.000000001, arr[:,:,11] == 180)
    mask = arr[:,:,colorlist.index('BNR')] < 500 # BNIR below 5%
    masked_arr = np.where(mask[:,:,None],np.nan, arr)
    makeTif_np_to_matching_tif(masked_arr, vrtPath, f"{vrtPath.split('.')[0]}_watermask.tif", gdalType=gdal.GDT_Float32, bands=len(b))


def stackReader(path_to_stack, bands=False, era=False):
    """Reads-in a raster stacks and returns a 3D numpy array of that array.
    Optionally, a list with band names will be returned

    Args:
        path_to_stack (str): path to the stack.tif
        bands (bool): If True, a list with band names of stack wil be returned as well. (WORKS ONLY WITH ERA5.grib files for now!!!!)
        era (bool): If bands True and era True, the bands metadata is extracted in a different manner
    """
    conti = []
    if type(path_to_stack) != osgeo.gdal.Dataset:
        ds = gdal.Open(path_to_stack)
    else:
        ds = path_to_stack
    ds = checkPath(path_to_stack)
    bandCount = ds.RasterCount
    if bands:
        bandsL = []
        if bandCount > 1:
            for b in range(bandCount):
                conti.append(ds.GetRasterBand(b+1).ReadAsArray())
                if not era:
                    bandsL.append(ds.GetRasterBand(b+1).GetDescription())
                else:
                    bandsL.append(datetime.fromtimestamp(int(ds.GetRasterBand(b+1).GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
            return np.dstack(conti), bandsL
        else:
            conti.append(ds.GetRasterBand(1).ReadAsArray())
            if not era:
                bandsL.append(ds.GetRasterBand(1).GetDescription())
            else:
                bandsL.append(datetime.fromtimestamp(int(ds.GetRasterBand(1).GetMetadata()['GRIB_VALID_TIME']), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
            return np.dstack(conti), bandsL
    else:
        if bandCount > 1:
            for b in range(bandCount):
                conti.append(ds.GetRasterBand(b+1).ReadAsArray())
            return np.dstack(conti)
        else:
            return ds.GetRasterBand(1).ReadAsArray()
        

def export_intermediate_products(row_col_start, intermediate_aray, dummy_gt, dummy_proj, folder_out, filename, noData=None, typ='int', comp=False):
    '''
    intermediate_aray: array to be exported
    dummy_gt + dummy_proj: GetGeotransform() and GetProjection from a gdal.Open object that contains desired geoinformation
    folder_out: path to FOLDER, where intermediate product will be stored
    noData = a no data value can be assigned to the exported tif
    comp (bool): If True, tiff uses options=['COMPRESS=DEFLATE', 'TILED=YES']
    '''
    if not folder_out.endswith('/'):
        folder_out = folder_out + '/'

    row_start = int(row_col_start.split('_')[0])
    col_start = int(row_col_start.split('_')[1])
    
    typi = gdal.GDT_Int32
    if typ == 'float':
        typi = gdal.GDT_Float32
    if comp:
        out_ds = gdal.GetDriverByName('GTiff').Create(f'{folder_out}{filename}', 
                                                    intermediate_aray.shape[1], intermediate_aray.shape[0], 1, typi,
                                                    options=['COMPRESS=DEFLATE', 'TILED=YES'])
    else:    
        out_ds = gdal.GetDriverByName('GTiff').Create(f'{folder_out}{filename}', 
                                                    intermediate_aray.shape[1], intermediate_aray.shape[0], 1, typi)
    # change the Geotransform for each chip
    geotf = list(dummy_gt)
    # get column and rows from filenames
    geotf[0] = geotf[0] + geotf[1] * col_start
    geotf[3] = geotf[3] + geotf[5] * row_start
    #print(f'X:{geoTF[0]}  Y:{geoTF[3]}  AT {file}')
    out_ds.SetGeoTransform(tuple(geotf))
    out_ds.SetProjection(dummy_proj)
                
    out_ds.GetRasterBand(1).WriteArray(intermediate_aray)
    if noData != None:
        out_ds.GetRasterBand(1).SetNoDataValue(noData)
    del out_ds


def makePyramidsForTif(tif_path):
    ds = gdal.Open(tif_path)
    ds.BuildOverviews("AVERAGE", [2, 4, 8, 16, 32])
    ds = None
    print('pyramids created')


def warp_np_to_reference(arr, arr_tif_path, target_tif_path, noData=np.nan, resamp=gdal.GRA_Bilinear, output_path=None, band_names=False):
    """
    Warps a NumPy array to the spatial resolution, projection, and extent of a target tif. Therefore, a path to a tif that holds the geoinfo of the array
    must be provided. Returns the warped array, while export as tif is optional.

    Args:
        arr (numeric np.array): hols the data that should be warped
        arr_tif_path (str): path to the tif that holds the geoinformation of arr, on which basis the array will be warped
        Target_tif_path (_type_): the tif, to which properties the array will be warped
        noData (numeric): Value of array that represents the noData value. Defaults to np.nan.
        resamp (gdal.GRA_, optional): Algorythm that should be used for resampling. Defaults to gdal.GRA_Bilinear.
        output_path (str, optional): If provided, the warped array will be stored as tiff at this location. Defaults to None.
        band_names (list, optional): If provided, bands in output tif will get these names
    """
    # determine whether arr has more than one band  
    arrDim = len(arr.shape)

    # open the tif file associated with the array and extract geo metadata
    src_ds = checkPath(arr_tif_path)
    gt_src = src_ds.GetGeoTransform()
    proj_src = src_ds.GetProjection()
    cols_src = src_ds.RasterXSize
    rows_src = src_ds.RasterYSize

    # create an in-memory tif of array to warp it
    mem_drv = gdal.GetDriverByName('MEM')
    if arrDim == 2:
        src_mem = mem_drv.Create('', cols_src, rows_src, 1, gdal.GDT_Float32)
        src_mem.SetGeoTransform(gt_src)
        src_mem.SetProjection(proj_src)
        src_mem.GetRasterBand(1).WriteArray(arr)
        src_mem.GetRasterBand(1).SetNoDataValue(noData)
    else:
        src_mem = mem_drv.Create('', cols_src, rows_src, arr.shape[2], gdal.GDT_Float32)
        src_mem.SetGeoTransform(gt_src)
        src_mem.SetProjection(proj_src)
        for band in range(arr.shape[2]):
            src_mem.GetRasterBand(band + 1).WriteArray(arr[:,:,band])
            src_mem.GetRasterBand(band + 1).SetNoDataValue(noData)

    # open target tif to extract spatial info
    ref_ds = checkPath(target_tif_path)
    gt_ref = ref_ds.GetGeoTransform()
    proj_ref = ref_ds.GetProjection()
    cols_ref = ref_ds.RasterXSize
    rows_ref = ref_ds.RasterYSize

    # warp it
    warped_ds = gdal.Warp('', src_mem,
              format='MEM',
              dstSRS=proj_ref,
              xRes=gt_ref[1],
              yRes=abs(gt_ref[5]),
              outputBounds=(
                  gt_ref[0],
                  gt_ref[3] + gt_ref[5] * rows_ref,
                  gt_ref[0] + gt_ref[1] * cols_ref,
                  gt_ref[3]
              ),
              resampleAlg=resamp)
    

    # export if path provided
    if output_path:
    
        if arrDim == 2:
            out_ds = gdal.GetDriverByName('GTiff').Create(output_path or '', cols_ref, rows_ref, 1, gdal.GDT_Float32)
            out_ds.SetGeoTransform(gt_ref)
            out_ds.SetProjection(proj_ref)
        else:
            out_ds = gdal.GetDriverByName('GTiff').Create(output_path or '', cols_ref, rows_ref, arr.shape[2], gdal.GDT_Float32)
            out_ds.SetGeoTransform(gt_ref)
            out_ds.SetProjection(proj_ref)

        # Return array
    if arrDim == 2:
        warped_array = warped_ds.GetRasterBand(1).ReadAsArray()
        if output_path:
            out_ds.GetRasterBand(1).WriteArray(warped_array)
            out_ds.GetRasterBand(1).SetNoDataValue(noData)
            if band_names:
                out_ds.GetRasterBand(1).SetDescription(band_names[0])
    else:
        warpL = []
        for band in range(arr.shape[2]):
            warpL.append(warped_ds.GetRasterBand(band + 1).ReadAsArray())
        warped_array = np.dstack(warpL)
        if output_path:
            for band in range(arr.shape[2]):
                out_ds.GetRasterBand(band + 1).SetNoDataValue(noData)
                out_ds.GetRasterBand(band + 1).WriteArray(warped_array[:,:,band])
            if band_names:
                for idx, bname in enumerate(band_names):
                    out_ds.GetRasterBand(idx + 1).SetDescription(bname)

    return warped_array

def assert_same_length(*lists): # * accepts any number of positional arguments and packs them into a tuple
    lengths = list(map(len, lists))
    if len(set(lengths)) != 1:
        raise ValueError(f"Lists have different lengths: {lengths}")
    
def getSpatRefRas(layer):
    # check type of layer
    if type(layer) is gdal.Dataset:
        SPRef = osr.SpatialReference()
        SPRef.ImportFromWkt(layer.GetProjection())

    elif type(layer) is str:
        lyr   = gdal.Open(layer)
        SPRef = osr.SpatialReference()
        SPRef.ImportFromWkt(lyr.GetProjection())

    return(SPRef)

def getSpatRefVec(layer):

    # check the type of layer
    if type(layer) is ogr.Geometry:
        SPRef   = layer.GetSpatialReference()

    elif type(layer) is ogr.Feature:
        lyrRef  = layer.GetGeometryRef()
        SPRef   = lyrRef.GetSpatialReference()

    elif type(layer) is ogr.Layer:
        SPRef   = layer.GetSpatialRef()

    elif type(layer) is ogr.DataSource:
        lyr     = layer.GetLayer(0)
        SPRef   = lyr.GetSpatialRef()

    elif type(layer) is str:
        lyrOpen = ogr.Open(layer)
        lyr     = lyrOpen.GetLayer(0)
        SPRef   = lyr.GetSpatialRef()

    return(SPRef)

def RasterKiller(raster_path):
    if os.path.isfile(raster_path):
        os.remove(raster_path)

############################ FROM RSS
def get_query(start_date, end_date, wkt_bbox, stac_geoparquet):
    """Generate SQL query for filtering STAC items by date and geometry."""
    sql_where = f'(("datetime" BETWEEN \'{start_date}T00:00:00Z\' AND \'{end_date}T23:59:59Z\')) AND (ST_Intersects(geometry, ST_GeomFromText(\'{wkt_bbox}\')) )'
    sql_query = f"SELECT * EXCLUDE(geometry),ST_AsWKB(geometry) as geometry FROM read_parquet('{stac_geoparquet}', union_by_name=False) WHERE {sql_where}"
    return sql_query

def add_sas_token(item, sas_token):
    """Add SAS token to all asset URLs in a STAC item."""
    for _, asset in item.assets.items():
        asset.href = f"{asset.href}?{sas_token}"
    return item