# Overview Evapotranspiration estimation

## Input Data
### Spatial extent: Germany, e.g. the bounding box used in ipynb, or bbox = [5.592041, 47.129951, 15.26001, 55.09723] 
### --> we use a tiling system at the moment, that is sth. for discussion

- Sentinel-3: LST with acquisition time, ViewingZenithAngle, ViewingAzimuthAngle

- Sentinel-2 9-day composites of bands [2,3,4,5,6,7,8,8A,11,12] --> ['BLU', 'GRN', 'RED', 'BNR', 'NIR', 'RE1', 'RE2', 'RE3',  'SW1', 'SW2'] (int)
 --> the filename of each composite must contain the date of the middle of the 9-day compositing time frame, e.g. composite of observations from 2017-12-01 
      until 2017-12-09 --> filename xxx_20171205.tif. Furthermore, if composites are stacks of the bands, the color (list above) should be present in the bandname

- ERA 5 datasets: ["2m_dewpoint_temperature", "2m_temperature", "surface_pressure", "100m_u_component_of_wind", "100m_v_component_of_wind", 
                        "total_column_water_vapour", "geopotential", "surface_solar_radiation_downward_clear_sky"]
                        all hourly from 2018-2025 (geopotential only one image)

- DEM: ["COPERNICUS_30"] and derived slope, aspect (float)
- Latitude and Longitude rasters @20m (float) (I would create them after we settled on tiling scheme)
- Thuenen agricultural mask for Germany for 2018-2025 (https://eodata.thuenen.de/collections/crop-type-map-latest); already downloaded, have to be 'warped' to Sentinel-2 (preproccesing)


## Preprocessing
- create a tiling system as processing across Germany probably too computaionally heavy
- Sentinel-2 compositing
- DEM (+derivates), Lat, Long, Thuenen all coregistered (warped) to Sentinel-2, same spatial extent and resolution (20m) (create only once?)
- Sentinel-3 compositing (0° threshold, VZA, acquisition time, 2m ERA-5 air temperature)
- Sentinel-3 sharpening (maybe including daily incidence calculation)
- maybe add biophysical parameter derivation
- Sentinel-3 will match other rasters after sharpening

## Apply model
- the model will be run per Sentinel-2 composite per spatial tile (or for the entirety of Germany if computing power can handle it! The sharpening appears to be the bottleneck)
- starting from the data of the composite, all Sentinel-3 LST images that fall within 9day threshold (+/- 4 days of composite date) will be sharpened
- 

## Questions transfer
- so far, we use the FORCE algorithm (https://force-eo.readthedocs.io/en/latest/) on our in-house Sentinel-2 datacube and our whole pipelin is based on its tiling scheme
- collecting files based on filenames (search scheme based on .split might be arbitrary/messy)
- how much data could be stored, e.g. files that don't change per iteration, e.g. DEM, SLope,




## Package list
numpy, osgeo, datetime, re, time, pandas, pvlib, scipy, collections, math, multiprocessing, sklearn, sys, numba, pyproj, pypro4sail, skyimage, rasterio