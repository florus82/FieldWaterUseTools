import os
import osgeo
from osgeo import gdal

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
