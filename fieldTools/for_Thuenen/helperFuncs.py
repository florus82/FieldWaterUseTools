import os
import numpy as np
import osgeo
from osgeo import gdal, ogr, osr
import re
import xml.etree.ElementTree as ET
import geopandas as gpd
import pandas as pd
import rasterio
import cv2
from scipy import ndimage
from datetime import datetime, timezone
from tqdm import tqdm
import xarray as xr
import rioxarray
from datetime import datetime
import random


# for IACS masking
EXCLUDE_LIST = ['',
 '(Beta-)Rübensamenvermehrung',
 'Alle anderen Flächen (keine LF)',
 'Baumschulen, nicht für Beerenobst',
 'Bestockte Rebfläche',
 'Erosionsschutzstreifen',
 'Forstflächen (Waldbodenflächen)',
 'Gewässerschutzstreifen',
 'Grassamenvermehrung',
 'Haus- und Nutzgärten',
 'KUP lt. Direktzahlungendurchführungsverordnung',
 'KUP lt. Direktzahlungendurchführungsverordnung (keine ÖVF)',
 'KUP lt. GAPDZV',
 'N. LNF, n. Art. 32(2b(i)) der VO(EG) Nr.1307/2013 beihilfef. Fl.',
 'Nicht landwirt. Fl. In der Verfügungsgewalt des Antragstellers, die gem. § 15 (1) DirektZahlDurchfG als umweltsensibles Dauergrünland bestimmt worden sind',
 'Nicht landwirt. Fl. infolge Genehmigung DGL Umwandlung',
 'Nicht landwirtschaftliche, aber nach Art. 32(2b (i)) der VO (EG) Nr. 1307/2013 beihilfefähige Fläche',
 'Nicht landwirtschaftliche, aber nach Art. 32(2b (i)) der VO (EG) Nr. 1307/2013 beihilfefähige Fläche (Naturschutzflächen, die 2008 noch beihilfefähig waren)',
 'Nicht landwirtschaftliche, aber nach §11 (1) Nr.3 Bst. a) bb) der GAPDZV förderfähige Fläche (Infolge Anwendung der Wasserrahmenrichtlinie)',
 'Nicht landwirtschaftliche, aber §11 (1) Nr.3 Bst. c) der GAPDZV förderfähige Fläche (Aufforstungsverpflichtung nach VO 1257/1999 oder VO (EG) Nr. 1698/2005 oder VO 1305/2013 oder VO 2021/2115 oder bei Eingehung damit in Einklang stehender öffentlich',
 'None',
 'Pufferstreifen ÖVF DGL',
 'Schonstreifen',
 'Streifen am Waldrand (ohne Produktion) ÖVF',
 'Ufervegetation ÖVF',
 'Unbefestigte Mieten-, Stroh-, Futter und Dunglagerplätze auf AL',
 'Unbefestigte Mieten-, Stroh-, Futter und Dunglagerplätze auf DGL',
 'Unbestockte Rebfläche',
 'Vorübergehende, unbefestigte Mieten-, Stroh-, Futter und Dunglagerplätze auf AL',
 'Vorübergehende, unbefestigte Mieten-, Stroh-, Futter und Dunglagerplätze auf DGL',
 'Weihnachtsbäume',
 'Wildäsungsfläche',
 'afforestation_reforestation',
 'alle anderen Flächen (keine LF)',
 'aufgeforstete Dauergrünlandflächen, weder nach 1257/99 oder VO (EG) Nr. 1698/2005  1305/2013oder VO (EU) Nr.1305/2013',
 'aufgeforstete Flächen (VO1257/1999, 1698/2005, 1305/2013)',
 'greenhouse_foil_film',
 'nach VO 1257/1999 oder VO (EG) Nr. 1698/2005 oder VO 1305/2013 aufgeforstete Flächen',
 'not_known_and_other',
 'nurseries_nursery',
 'tree_wood_forest',
 'unmaintained',
 'vorübergehend unbefestigte Mieten-, Stroh-, Futter oder Dunglagerplätze auf DGL',
 'vorübergehende, unbefestigte Mieten-, Stroh-, Futter oder Dunglagerplätze auf AL',
 'Hecke/Knick (CC-LE)',
 'Hecke/Knick (LE)', 
 'Feldgehölz (CC-LE)', 
 'Feldgehölz (LE)', 
 'Feuchtgebiet (CC-LE)', 
 'Feldrain (LE)', 
 'Feldrain (CC-LE)', 
 'Fels-, Steinriegel, naturversteinte Fläche (CC-LE)', 
 'Trocken-, Natursteinmauer, Lesesteinwall (CC-LE)', 
 'Tümpel, Söll, Doline (CC-LE)',
 'Baumreihe (CC-LE)',
 'Graben (LE)',
 'nicht landw. Fläche']

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

class AI4BNormal_S2(object):
    """
    class for Normalization of images, per channel, in format CHW 
    """
    def __init__(self):

        self._mean_s2 = np.array([5.4418573e+02, 7.6761194e+02, 7.1712860e+02, 2.8561428e+03 ]).astype(np.float32) 
        self._std_s2  = np.array( [3.7141626e+02, 3.8981952e+02, 4.7989127e+02 ,9.5173022e+02]).astype(np.float32) 

    def __call__(self,img):

        temp = img.astype(np.float32)
        temp2 = temp.T
        temp2 -= self._mean_s2
        temp2 /= self._std_s2

        temp = temp2.T
        return temp

def loadVRTintoNumpyAI4(vrtPath, applyNormalizer=True):
    '''vrtPath: path in which vrts are stored
        vrts will be loaded into numpy array and normalized (for Sentinel-2 10m bands!!!!!)'''
    vrtFiles = [file for file in getFilelist(vrtPath, '.vrt') if 'Cube' not in file]
    vrtFiles = sortListwithOtherlist([int(vrt.split('_')[-1].split('.')[0]) for vrt in vrtFiles], vrtFiles)[-1]
    bands = []

    for vrt in vrtFiles:
        ds = gdal.Open(vrt)
        bands.append(ds.GetRasterBand(1).ReadAsArray())
    cube = np.dstack(bands)
   
    data_cube = np.transpose(cube, (2, 0, 1))
    reshaped_cube = data_cube.reshape(4, 6, ds.RasterYSize, ds.RasterXSize)
    normalizer = AI4BNormal_S2()
    if applyNormalizer:
        return normalizer(reshaped_cube)
    else:
        return reshaped_cube

def RasterKiller(raster_path):
    if os.path.isfile(raster_path):
        os.remove(raster_path)

def reduce_forceTSA_output_to_validmonths(path_to_forceoutput, start_month_int, end_month_int):
    '''path_to_forceoutput: path of stored force output (quite likely you want the folder in which all tile folders are)
    start_month_int & end_month_int: e.g. 3 for march and 8 for August
    Please note: The filter will look for the YYYYMMDD characters that come right before .tif
    '''
    # get rid of force output that is not needed -> months outside of growing season that do not exist in AI4Boundaries
    files = getFilelist(path_to_forceoutput, '.tif', deep=True)

    filesToKill = [f for f in files if int(f.split('-')[-1].split('.')[0]) not in [i for i in range(start_month_int, end_month_int + 1, 1)]]

    for file in filesToKill:
        RasterKiller(file)

    return list(filter(lambda item: item not in filesToKill, files))


def force_order_Colors_for_VRT(list_of_forcefiles, list_of_colors, list_of_days):
    '''list_of_forcefiles: e.g. output from reduce_force_to_validmonths
        will return a list that orders the input list to blue, green, red, ir independently from tiles and dates'''
    tiles = list(set([file.split('output/')[-1].split('/')[2].split('/')[0] for file in list_of_forcefiles]))
    tilefilesL = []
    for color in list_of_colors:
        for day in list_of_days:
            for tile in tiles:
                for file in list_of_forcefiles:
                    if tile in file and color in file and day in file and tile in file:
                        tilefilesL.append(file)


    if len(tilefilesL) == len(list_of_colors) * len(list_of_days) * len(tiles):
        return [tilefilesL[i:i+len(tiles)] for i in range(0, len(tilefilesL), len(tiles))]
    else:
        raise ValueError('length of colors, days and files do not line up')

def getFORCExyRangeName(tiles):
    '''take a list of subsetted FORCE Tile names in the Form of X0069_Y0042 and returns a string to be used as filename 
    that gives X and Y range ,e.g. Force_X_from_68_to_69_Y_from_42_to_42'''
    X = [int(tile.split('_')[0][-2:]) for tile in tiles]
    Y = [int(tile.split('_')[1][-2:]) for tile in tiles]

    return f'Force_X_from_{min(X)}_to_{max(X)}_Y_from_{min(Y)}_to_{max(Y)}'

def get_forcetiles_range(list_of_forcefiles):
    '''list_of_forcefiles: e.g. output from reduce_force_to_validmonths
    creats a string that indicates X and Y extremes from list_of_forcefiles'''
    return list(set([re.search(r'X\d{4}_Y\d{4}', tile).group() for tile in list_of_forcefiles]))

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

def force_to_vrt(list_of_forcefiles, ordered_forcetiles, vrt_out_path, pyramids=False, bandnames=False):
    '''list_of_forcefiles: e.g. output from reduce_force_to_validmonths
        ordered_forcetiles: e.g output from getCOLORSinOrderFORCELIST (single=False)
        vrt_out_path: path where .vrt files will be created (there will be more than one to account for all the bands)
        pyramids: if set to True, pyramids will be created (might be very very large!!)'''
    
    # tiles = list(set([file.split('output/')[-1].split('/')[1].split('/')[0] for file in list_of_forcefiles]))
    force_folder_name = getFORCExyRangeName(get_forcetiles_range(list_of_forcefiles))
    if not vrt_out_path.endswith('/'):
        vrt_out_path = vrt_out_path + '/'
    outDir = f'{vrt_out_path}{force_folder_name}/'
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print(outDir)
        vrts = []
        for i in range(len(ordered_forcetiles)):
            vrt_name = f'{outDir}{force_folder_name}_{str(i)}.vrt'
            vrt = gdal.BuildVRT(vrt_name, ordered_forcetiles[i], separate = False)
            vrt = None

            # make paths in vrts relative
            convertVRTpathsTOrelative(vrt_name)
            vrts.append(vrt_name)

        # set optionally bandnames    
        if bandnames:
            for idz, bname in enumerate(np.repeat(bandnames,int(len(ordered_forcetiles) / len(bandnames))).tolist()):  
                print(f'{outDir}{force_folder_name}_{str(idz)}.vrt')
                vrt = gdal.Open(f'{outDir}{force_folder_name}_{str(idz)}.vrt', gdal.GA_Update)  # VRT must be writable
                band = vrt.GetRasterBand(1)
                band.SetDescription(bname)
                vrt = None
        print('single vrts created')
        
        nums = [int(vrt.split('_')[-1].split('.')[0]) for vrt in vrts]
        vrts_sorted = sortListwithOtherlist(nums, vrts)[-1]
        print('paths in vrts made relative')
        
        vrt = gdal.BuildVRT(f'{outDir}{force_folder_name}_Cube.vrt', vrts_sorted, separate = True)
        vrt = None
        if bandnames:
            # set vrt band names
            vrt = gdal.Open(f'{outDir}{force_folder_name}_Cube.vrt', gdal.GA_Update)  # VRT must be writable
            for idz, bname in enumerate(np.repeat(bandnames,int(len(ordered_forcetiles) / len(bandnames))).tolist()): 
                band = vrt.GetRasterBand(1+idz)
                band.SetDescription(bname)
            vrt = None
        # convertVRTpathsTOrelative(f'{outDir}{force_folder_name}_Cube.vrt')
        print('overlord vrt created')
        if pyramids:
            # build pyramids
            vrtPyramids(f'{outDir}{force_folder_name}_Cube.vrt')
            print('VRT created with pyramids')
    else:
        print('Vrt might already exist - please check!!')

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

def dirfinder(path):
    """ returns a list with all directory names within a folder

    Args:
        path (str): str path to folder that will be searched for directories

    Returns:
        list: list of directory names (str)
    """
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

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

############################ IACS rasterization
# create dataset in memory using geotransform specified in ref_pth
def create_mem_ds(ref_pth, n_bands):
        drvMemR = gdal.GetDriverByName('MEM')
        ds = gdal.Open(ref_pth)
        mem_ds = drvMemR.Create('', ds.RasterXSize, ds.RasterYSize, n_bands, gdal.GDT_Float32)
        mem_ds.SetGeoTransform(ds.GetGeoTransform())
        mem_ds.SetProjection(ds.GetProjection())
        return mem_ds

# create copy
def copy_mem_ds(pth, mem_ds):
        copy_ds = gdal.GetDriverByName("GTiff").CreateCopy(pth, mem_ds, 0, options=['COMPRESS=LZW'])
        copy_ds = None

######################
# multi-taks labels from boundaries
def get_boundary(label, kernel_size = (2,2)):
    tlabel = label.astype(np.uint8)
    temp = cv2.Canny(tlabel,0,1)
    tlabel = cv2.dilate(
        temp,
        cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            kernel_size),
        iterations = 1)
    tlabel = tlabel.astype(np.float32)
    tlabel /= 255.
    return tlabel

def get_distance(label):
    tlabel = label.astype(np.uint8)
    dist = cv2.distanceTransform(tlabel,
                                 cv2.DIST_L2,
                                 0)

    # get unique objects
    output = cv2.connectedComponentsWithStats(label, 4, cv2.CV_32S)
    num_objects = output[0]
    labels = output[1]

    # min/max normalize dist for each object
    for l in range(num_objects):
        dist[labels==l] = (dist[labels==l]) / (dist[labels==l].max())

    return dist

def get_crop(image, kernel_size = (3,3)):

    im_floodfill = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # floodfill
    cv2.floodFill(im_floodfill, mask, (0,0), 1);

    # invert
    im_floodfill = cv2.bitwise_not(im_floodfill)

    # kernel size
    kernel = np.ones(kernel_size, np.uint8)

    # erode & dilate
    img_erosion = cv2.erode(im_floodfill, kernel, iterations=1)
    return cv2.dilate(img_erosion, kernel, iterations=1) - 254

# 00_polygons_to_lines.py
def polygons_to_lines(path_to_polygon, path_to_lines_out, categories=None, category_col=None):
    
    if not os.path.exists(path_to_lines_out):
        # Load polygons
        if path_to_polygon.endswith(('.geoparquet', '.parquet')):
            gdf = gpd.read_parquet(path_to_polygon)
        else:
            gdf = gpd.read_file(path_to_polygon)
        
        # Filter categories if provided
        if categories is not None and category_col is not None:
            gdf = gdf[~gdf[category_col].isin(categories)]
        
        # Keep only valid geometries
        gdf = gdf[gdf.is_valid]
        
        # Drop duplicate geometries
        gdf = gdf.drop_duplicates(subset='geometry')
        
        # Convert polygons to lines
        gdf['geometry'] = gdf.geometry.boundary
        
        # Drop any existing 'fid' column to avoid UNIQUE constraint
        if 'fid' in gdf.columns:
            gdf = gdf.drop(columns=['fid'])
        
        # Reset index to ensure unique IDs
        gdf = gdf.reset_index(drop=True)
        
        # Construct layer name
        layer_name = path_to_polygon.split('/')[-1].split('.')[0]
        
        # Save to GeoPackage using Fiona backend (safer for new layers)
        gdf.to_file(path_to_lines_out, driver='GPKG', layer=layer_name)
        
    else:
        print(f'Polylines for polygon {path_to_polygon} already exists!!!')


# 01_rasterize_line_feats
def rasterize_lines(path_to_lines, path_to_extent_raster, path_to_rasterlines_out, all_touch=True, dilate=False, dmatrix = (2,2), dlevel=1):

    if not os.path.exists(path_to_rasterlines_out):
        ##### open field vector file
        field = ogr.Open(path_to_lines)
        field_lyr = field.GetLayer(0)

        ds = gdal.Open(path_to_extent_raster)
        target_ds = gdal.GetDriverByName('GTiff').Create(path_to_rasterlines_out, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
        target_ds.SetGeoTransform(ds.GetGeoTransform())
        target_ds.SetProjection(ds.GetProjection())

        if all_touch:
             opti = ["ALL_TOUCHED=TRUE"]
        else:
             opti = ["ALL_TOUCHED=FALSE"]
        gdal.RasterizeLayer(target_ds, [1], field_lyr, burn_values=[1], options = opti)
        target_ds = None

        if dilate:
            ds_l = gdal.Open(path_to_rasterlines_out)
            edge = ds_l.GetRasterBand(1).ReadAsArray()
            edge = cv2.dilate(edge, np.ones(dmatrix, np.uint8), dlevel)
            storPath = path_to_rasterlines_out.split('.')[0] + '_dilated.tif'
            target_ds = gdal.GetDriverByName('GTiff').Create(storPath, ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Byte)
            target_ds.SetGeoTransform(ds.GetGeoTransform())
            target_ds.SetProjection(ds.GetProjection())
            target_ds.GetRasterBand(1).WriteArray(edge)

            target_ds = None

    else:
        print(f'Rasterized lines for {path_to_lines} already exists!!!')


# 02_multitask_labels
def make_multitask_labels(path_to_rasterlines, path_to_mtsk_out):

    edge = cv2.imread(path_to_rasterlines, cv2.IMREAD_GRAYSCALE)
    crop = get_crop(edge)
    dist = get_distance(crop)
    edge = cv2.dilate(edge, np.ones((2,2), np.uint8), 1)

    label = np.stack([crop, edge, dist])
    mem_ds = create_mem_ds(path_to_rasterlines, 3)

    # write outputs to bands
    for b in range(3):
        mem_ds.GetRasterBand(b+1).WriteArray(label[b,:,:])

    # create physical copy of ds
    copy_mem_ds(path_to_mtsk_out, mem_ds)

# 03 create a crop mask
def make_crop_mask(path_to_polygon, path_to_extent_raster, path_to_mask_out, path_to_rasterized_lines=False,\
                   all_touch=True, categories=None, category_col=None, burn_col = None):
    
    if not os.path.exists(path_to_mask_out):
        
        #### open field vector file
        if path_to_polygon.endswith(('.geoparquet', '.parquet')):
            field_gpd = gpd.read_parquet(path_to_polygon)
        else:
            field_gpd = gpd.read_file(path_to_polygon)

        if burn_col == "id_0":
            field_gpd = field_gpd.reset_index(drop=True)
            field_gpd["id_0"] = field_gpd.index + 1

        # Exclude specified categories if provided
        if categories is not None and category_col is not None:
            field_gpd = field_gpd[~field_gpd[category_col].isin(categories)]

        # Keep only valid geometries
        field_gpd = field_gpd[field_gpd.is_valid]
        field_gpd = field_gpd[field_gpd.geometry.notnull()]
        field_gpd["geometry"] = field_gpd.buffer(0)

        # Drop duplicate geometries
        field_gpd = field_gpd.drop_duplicates(subset='geometry')

        # convert field_gpd to vector layer for rasterization
        ogr_ds = ogr.GetDriverByName('Memory').CreateDataSource('')
        srs = ogr.osr.SpatialReference()

        # account for potential projection mismatch
        ds = gdal.Open(path_to_extent_raster)
        raster_wkt = ds.GetProjection()
        field_gpd = field_gpd.to_crs(raster_wkt)

        srs.ImportFromWkt(raster_wkt)

        field_lyr = ogr_ds.CreateLayer("field_layer", srs, ogr.wkbPolygon)
        
        # if burn_col:
        #     # Create fields (attributes) in the OGR layer 
        #     for col_name, dtype in zip(field_gpd.columns, field_gpd.dtypes):
        #         if col_name == field_gpd.geometry.name:
        #             continue  # skip geometry column

        #         if np.issubdtype(dtype, np.integer):
        #             field_type = ogr.OFTInteger64
        #         elif np.issubdtype(dtype, np.floating):
        #             field_type = ogr.OFTReal
        #         else:
        #             field_type = ogr.OFTString

        #         field_defn = ogr.FieldDefn(col_name, field_type)
        #         field_lyr.CreateField(field_defn)
        if burn_col:
            field_defn = ogr.FieldDefn(burn_col, ogr.OFTInteger64)
            field_lyr.CreateField(field_defn)

        # Convert geometries
        layer_defn = field_lyr.GetLayerDefn()
        for _, row in field_gpd.iterrows():
        
            # Validate geometry
            if row.geometry is None or row.geometry.is_empty:
                print("Skipping empty geometry.")
                continue

            # Convert to OGR geometry
            geom = ogr.CreateGeometryFromWkb(row.geometry.wkb)
            if geom is None:
                print("Invalid geometry encountered, skipping.")
                continue

            feature = ogr.Feature(layer_defn)
            feature.SetGeometry(geom)

            if burn_col:
                if burn_col:
                    value = row[burn_col]
                    if pd.notnull(value):
                        feature.SetField(burn_col, int(value))

                # for col_name in field_gpd.columns:
                #     if col_name != field_gpd.geometry.name:
                #         value = row[col_name]
                #         if pd.notnull(value):
                #             feature.SetField(col_name, value)
            field_lyr.CreateFeature(feature)
            feature = None  # Free memory

        if burn_col:
            export_dt = gdal.GDT_UInt32
        else:
            export_dt = gdal.GDT_Byte
   
        target_ds = gdal.GetDriverByName('GTiff').Create(path_to_mask_out, ds.RasterXSize, ds.RasterYSize, 1, export_dt)
        target_ds.SetGeoTransform(ds.GetGeoTransform())
        target_ds.SetProjection(ds.GetProjection())
        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(0)
        band.Fill(0)

        # this choice appears to have no effect when rasterizing the polygons
        if all_touch:
            opti = ["ALL_TOUCHED=TRUE"]
        else:
            opti = ["ALL_TOUCHED=FALSE"]

        if burn_col:
            opti.append(f"ATTRIBUTE={burn_col}")
            gdal.RasterizeLayer(target_ds, [1], field_lyr, options = opti)
        else:
            gdal.RasterizeLayer(target_ds, [1], field_lyr, burn_values=[1], options = opti)
        target_ds = None
    else:
        print(f'Mask for {path_to_polygon} already exists!!!')

    # mask the output with rasterized lines to clean up
    if path_to_rasterized_lines:
        path_linecrop_out = path_to_mask_out.split('.')[0] + '_linecrop.tif' # + '_lines_touch_' + path_to_rasterized_lines.split('_')[-1].split('.')[0]
        if not os.path.exists(path_linecrop_out):
            mask_ds = gdal.Open(path_to_mask_out)
            mask = mask_ds.GetRasterBand(1).ReadAsArray()
            lines_ds  = gdal.Open(path_to_rasterized_lines)
            lines = lines_ds.GetRasterBand(1).ReadAsArray()
            
            mask[np.where(lines == 1)] = 0

            target_ds = gdal.GetDriverByName('GTiff').Create(path_linecrop_out, mask_ds.RasterXSize, mask_ds.RasterYSize, 1, gdal.GDT_Byte)
            target_ds.SetGeoTransform(mask_ds.GetGeoTransform())
            target_ds.SetProjection(mask_ds.GetProjection())
            target_ds.GetRasterBand(1).WriteArray(mask)
            del target_ds
        else:
            print(f'Mask for {path_to_polygon} in combination with {path_to_rasterized_lines} already exists!!!')
    else:
        pass


def polygon_distance_normalized(path_to_uniqueIDfields, outPath=False, background_val=False):
    """
    Compute per-polygon normalized distance-to-border, treating *any other polygon*
    or background as border (i.e. field edges even if neighboring another field).

    path_to_uniqueIDfields: raster with integer IDs
    outPath: float raster with normalized distances per polygon (0 outside, (0,1] inside)
    background_val: value that is not a fieldID
    """


    if isinstance(path_to_uniqueIDfields, str):
        with rasterio.open(path_to_uniqueIDfields) as src:
            arr = src.read(1)
            profile = src.profile.copy()
            arr = np.asarray(arr)
    else:
        arr = path_to_uniqueIDfields
    out = np.zeros_like(arr, dtype=np.float32)

    if not background_val:
        background_val = 0
    unique_ids = np.unique(arr)
    unique_ids = unique_ids[unique_ids != background_val]  # drop background

    # --- Process each polygon separately ---
    for pid in unique_ids:
        # mask of current polygon
        mask = (arr == pid)

        # compute distance to the nearest *outside* pixel (anything not pid)
        # distance_transform_edt expects True inside, False outside
        dist = ndimage.distance_transform_edt(mask)

        # normalize to [0,1] within that polygon
        maxd = dist[mask].max()
        if maxd > 0:
            out[mask] = (dist[mask] / maxd).astype(np.float32)
        else:
            out[mask] = 1.0  # degenerate polygon (single pixel)

    # background = 0
    out[arr == 0] = 0.0

    if outPath:
        # update profile
        profile.update(dtype=np.float32, count=1, compress='lzw')
        with rasterio.open(outPath, "w", **profile) as dst:
            dst.write(out, 1)
    else:
        return out


def get_distance_raster(path_to_object, outPath):
    """Create a distance raster from a binary raster (distance to value 1)

    Args:
        path_to_object (str): path to binary raster
        outPath (str): path to output.tif on disc where distance raster will be stored
    """
    if not os.path.exists(outPath):
        # Open the input
        src_ds = gdal.Open(path_to_object)
        # Create output raster
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(
            outPath,
            src_ds.RasterXSize,
            src_ds.RasterYSize,
            1,
            gdal.GDT_Float32
        )
        dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
        dst_ds.SetProjection(src_ds.GetProjection())

        # Compute distance raster
        gdal.ComputeProximity(
            src_ds.GetRasterBand(1),
            dst_ds.GetRasterBand(1),
            options=["VALUES=1", "DISTUNITS=GEO"]
        )

        dst_ds = None

        print(f'Distance raster for {path_to_object} created')
    else:
        print(f'distance raster for {path_to_object} already exists!!!')