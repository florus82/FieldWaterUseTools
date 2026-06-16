from osgeo import gdal
import os
import numpy as np
import re
from FieldWaterUseTools.FuncBox.Misc import getFilelist, RasterKiller, vrtPyramids, convertVRTpathsTOrelative, sortListwithOtherlist, RasterKiller



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


def check_forceTSI_compositionDates(listOfFORCEoutput):
    """_summary_

    Args:
        listOfFORCEoutput (list_of_strings): list with paths to tif files from FORCE TSI output
    """
    fatal_check = 0
    date_list = []
    tiles = get_forceTSI_output_Tiles(listOfFORCEoutput)
    for tile in tiles:
        date_list.append((get_forceTSI_output_DOYS([file for file in listOfFORCEoutput if tile in file])))
    for i in range(0,len(date_list)-1):
        if date_list[i] == date_list[i + 1]:
            continue
        else:
            fatal_check = 1
    if fatal_check:
        print('the doys of composites across tiles is not equal - Better check!!!!! - No date list returned!!!!!!')
    else:
        print('all dates of composites are the same :)')
        return date_list[0]
    

def get_forceTSI_output_DOYS(listOfFORCEoutput):
    '''
    Will return a sorted list of unique DOYs in format YYYYMMDD. Please note, that this only works with files in FORCE naming convention, where
    the date will be used that is at the end of the filename (YYYYMMDD.tif)
    listOfFORCEoutput: list with paths to tif files from FORCE TSI output
    '''
    return sorted(list(set([re.search(r'(\d{4})(\d{2})(\d{2})\.tif$', file)[0].split('.tif')[0] for file in listOfFORCEoutput])))

def get_forceTSI_output_Tiles(listOfFORCEoutput):
    """
    Will return a sorted list of unique Tiles Ids(e.g. 'X0057_Y0044')

    Args:
        listOfFORCEoutput (list_of_strings): list with paths to tif files from FORCE TSI output
    """
    return sorted(list(set([re.search(r'X\d{4}_Y\d{4}',file)[0] for file in listOfFORCEoutput])))

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