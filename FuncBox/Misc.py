import os
import osgeo
import xml.etree.ElementTree as ET
import numpy as np
import random

from osgeo import gdal
from datetime import datetime, timezone


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