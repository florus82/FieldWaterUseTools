The scripts for creating the training dataset from FORCE and IACS data (01_Create_Training_Data.py), running the DL architecture (02_Run_DL_with_np.py) on this training dataset and then to make a prediction on with the resulting model (03_Prediction_GER.py) can be found under the “for_Thuenen” folder in my “FieldWaterUseTools” repo. There are also supporting scripts (tfcl folder + helperFuncs.py, helperFuncs_GPU.py) as well as a folder “FORCE” with parameter and supporting files.
The folder from the repo can be cloned (without the entire repo) via:
git clone --filter=blob:none --no-checkout https://github.com/florus82/FieldWaterUseTools.git
cd FieldWaterUseTools
git sparse-checkout init --cone
git sparse-checkout set fieldTools/for_Thuenen
git checkout main

The needed python environments for both scripts can be created from/downloaded:
01_Create_Training_Data.py			CreateTraining.yml (in this fodler)
02_Run_DL_with_np.py  docker image from fdiakogiannis/trchprosthesis_requirements:24.07-py (can be obtained via docker pull)
03_Prediction_GER.py  same docker images as above

In the FORCE folder, there are 2 parameter files. Each file produces the median composites for one year (specified via DATE_RANGE). The 2 prm files are set for the year 2017, while “2017_FBM_magic_S2.prm” only uses observations made by Sentinel-2, “2017_FBM_magic_S2_LS” also includes Landsat observation. 
The scripts are currently set to be used with force-magic-parameters, so that each tile defined in the first two rows is processed individually (as processing on HU server has been a bit unreliable in the past). Running force-magic-parameters combined with a .py script to check if processing of tiles has been successful and restart processing if needed can be accomplished with “sequence_fancy.sh” and “clean_up_failed_FORCE_down_BM.py”. However, the .sh file had to be adapted a bit (if this procedure is needed at all) if used outside of HU server settings. Furthermore, there is a folder “tilefiles” that contains .txt files with FORCE tiles for each state and the entirety of Germany (if the tilesystem is the same on the cluster as within HU compounds).