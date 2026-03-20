import os
import sys
import time
import requests
import openeo
import cdsapi



#origin = '/workspace/'
import sys
sys.path.append('/home/potzschf/repos/')
origin =  '/data/Aldhani/eoagritwin/'
#sys.path.append('/media/') # only needed to find FieldWaterUseTools on my machine

from FieldWaterUseTools.FuncBox.Misc import getFilelist, path_safe


# set masterpath for stored data
storPath_master = path_safe(f"{origin}et/test/")#('/place/to/store/porducts/')
storPath_S2_template = path_safe(f"{storPath_master}DEM/FORCE_TILES/DEM/")
storPath_S3_template = path_safe(f"{storPath_master}templates/S3_template.tif")
storPath_ERA5 = path_safe(f"{storPath_master}ERA5/")
path_to_geopot_raw = f"{storPath_ERA5}grib/geopotential/geopotential_unique.grib"

# set year and month for downloading S2 & S3 --> if we do the updates monthwise...
year = 2020
month = 4

########################################################################## templates of S2 and S3 images

###### S2 (this is actually a DEM reprojected and resampled to match S2 data obtained by the FORCE (https://force-eo.readthedocs.io/en/latest/
#           Furthermore, the entire image is cutted into smaller tiles (1500x1500). At the moment, we would proceed with this setup as it is not clear yet,
#           if the FORCE might become available)

# paths
# base_url = "https://box.hu-berlin.de"
# share_token = "5501a12dad894b01b505"

# # get all files in shared folder
# r = requests.get(
#     f"{base_url}/api/v2.1/share-links/{share_token}/dirents/",
#     params={"path": "/"},
# )
# r.raise_for_status()
# files = r.json()["dirent_list"]

# # download each file
# for f in files:
#     filename = f['file_name']
#     out_path_file = f"{storPath_S2_template}{filename}"

#     download_url = f"{base_url}/d/{share_token}/files/?p=/{filename}&dl=1"
#     file_r = requests.get(download_url, timeout=120)
#     file_r.raise_for_status()
    
#     with open(out_path_file, "wb") as fh:
#         fh.write(file_r.content)

# ###### S3

# # fake header
# headers = {
#     "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
# }
# # paths
# s3_temp_url = "https://box.hu-berlin.de/seafhttp/f/25328869da9b425cb2f6/?op=view"

# # download
# r = requests.get(s3_temp_url, headers=headers, stream=True, timeout=120)
# r.raise_for_status()
# with open(storPath_S3_template, "wb") as f:
#     f.write(r.content)


 
########################################################################## DEM

# if we go with the FORCE tiling scheme, we can just use the images from the template, otherwise:

# storPath_DEM = path_safe(f"{storPath_master}DEM/raw_tiles/")
# connection = openeo.connect("openeo.dataspace.copernicus.eu").authenticate_oidc()
# while len(getFilelist(storPath_DEM, '.tif')) < 99:
#     for i in range(0,11,1):
#         for j in [0,1]:# in range(0,10,1):
#             aoi = {'west': 5+i, 'south': 46+j, 'east': 6+i, 'north': 47+j}
#             storP = f"{storPath_DEM}DEM_GER_{j}_{i}.tif"
            
#             if os.path.exists(storP):
#                     t = time.localtime()
#                     ti = time.strftime("%H:%M:%S", t)
#                     print(f"already exists - next one at {ti}")

#             else:
#                 try:
#                     dem_cube = connection.load_collection(
#                                     "COPERNICUS_30",
#                                     spatial_extent = aoi,
#                                     bands=["DEM"]
#                                     )                               
#                     dem_cube.download(storP)

#                 except Exception as e:
#                     print(e)
#                     t = time.localtime()
#                     ti = time.strftime("%H:%M:%S", t)
#                     print(f"thrown at {ti}")
#                     continue


########################################################################## ERA5

client = cdsapi.Client()

dataset = "reanalysis-era5-single-levels"

variables = [             
        "2m_dewpoint_temperature", 
        "2m_temperature",
        "surface_pressure",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "total_column_water_vapour",
        "surface_solar_radiation_downward_clear_sky"]

for variable in variables:
    
    varPath = path_safe(f"{storPath_ERA5}grib/{variable}")
    storPath = f"{varPath}/{variable}_{year}_{month:02d}.grib"

    if os.path.exists(storPath):
        pass
    else:
        try:
            request = {
                "product_type": ["reanalysis"],
                "variable": [variable],
                "year": [year],
                "month": [5],
                "day": [
                    "01", "02", "03",
                    "04", "05", "06",
                    "07", "08", "09",
                    "10", "11", "12",
                    "13", "14", "15",
                    "16", "17", "18",
                    "19", "20", "21",
                    "22", "23", "24",
                    "25", "26", "27",
                    "28", "29", "30",
                    "31"
                ],
                "time": [
                    "00:00", "01:00", "02:00",
                    "03:00", "04:00", "05:00",
                    "06:00", "07:00", "08:00",
                    "09:00", "10:00", "11:00",
                    "12:00", "13:00", "14:00",
                    "15:00", "16:00", "17:00",
                    "18:00", "19:00", "20:00",
                    "21:00", "22:00", "23:00"
                ],
                "data_format": "grib",
                "download_format": "unarchived",
                "area": [56, 5, 47, 16]
            }

            target = storPath

            client.retrieve(dataset, request, target)
        
        except Exception as e:
            print(e)
            t = time.localtime()
            ti = time.strftime("%H:%M:%S", t)
            print(f"thrown at {ti}")
            continue

if os.path.exists(path_safe(path_to_geopot_raw)):
    pass
else:
    try:
        request = {
            "product_type": ["reanalysis"],
            "variable": ["geopotential"],
            "year": [2020],
            "month": [month],
            "day": ["01"],
            "time": ["13:00"],
            "data_format": "grib",
            "download_format": "unarchived",
            "area": [56, 5, 47, 16]
        }


        target = path_to_geopot_raw

        client.retrieve(dataset, request, target)

    except Exception as e:
        print(e)
        t = time.localtime()
        ti = time.strftime("%H:%M:%S", t)
        print(f"thrown at {ti}")
