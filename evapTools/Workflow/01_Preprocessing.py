# Sentinel-2 compositing and tiling


# Sentinel-3 compositing
# under the assumption that Sentinel-3 LST, VZA, VVA and acquisition time is ready to call.
- '/home/potzschf/repos/evapo_scripts/Guzinski/aux/ERA_5_2m_AIR_GEOPOT_for_LST_QA.py'
- '/home/potzschf/repos/evapo_scripts/Sentinel/py/make_masked_LST_stacks.py'


##### preprocessing that could be done once, if products could be stored:
# reproject/resample/align DEM with Sentinel-2 & derive slope, aspect
# calculate lat/lon
# prepare thuenen maps 
- '/home/potzschf/repos/evapo_scripts/Guzinski/aux/DEM_to_FORCE_and_OTHERS.ipynb' # also creates lat/lon
- '/home/potzschf/repos/evapo_scripts/Guzinski/aux/ERA_5_2m_AIR_GEOPOT_for_LST_QA.py'