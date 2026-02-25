# Overview Evapotranspiration estimation

## Input Data
### Spatial extent: Germany --> we use a tiling system at the moment, that is sth. for discussion

- Sentinel-3: LST, ViewingZenithAngle, ViewingAzimuthAngle

- Sentinel-2 9-day composites of bands [2,3,4,5,6,7,8,8A,11,12] --> ['BLU', 'GRN', 'RED', 'BNR', 'NIR', 'RE1', 'RE2', 'RE3',  'SW1', 'SW2'] (int)

- ERA 5 datasets: ["2m_dewpoint_temperature", "2m_temperature", "surface_pressure", "100m_u_component_of_wind", "100m_v_component_of_wind", 
                        "total_column_water_vapour", "geopotential", "surface_solar_radiation_downward_clear_sky"]

                        all hourly from 2018-2025 (geopotential only one image)

- DEM: ["COPERNICUS_30"] and derived slope, aspect (float)
- Latitude and Longitude rasters @30m (float)

- Thuenen agricultural mask for Germany for 2018-2025 (https://eodata.thuenen.de/collections/crop-type-map-latest)


## Preprocessing

- Sentinel-3 compositing
- Sentinel-3 sharpening (maybe including daily incidence calculation)
- maybe add biophysical parameter derivation
- Sentinel-2, DEM (+derivates), Lat, Long, Thuenen all coregistered (warped)
- Sentinel-3 will match other rasters after sharpening

## Apply model
- the model will be run per Sentinel-2 composite per spatial tile (or for the entirety of Germany if computing power can handle it! The sharpening appears to be the bottleneck)
- starting from the data of the composite, all Sentinel-3 LST images that fall within 9day threshold (+/- 4 days of composite date) will be sharpened
- 

## Issues with transfer
- so far, we use the FORCE algorithm (https://force-eo.readthedocs.io/en/latest/) on our in-house Sentinel-2 datacube and our whole pipelin is based on its tiling scheme
- collecting files based on filenames (search scheme based on .split might be arbitrary/messy)

## Package list
numpy, osgeo, datetime, re, time, pandas, pvlib, scipy, collections, math, multiprocessing, sklearn, sys, numba, pyproj, pypro4sail, skyimage, rasterio