# Overview

This is the repository that contains all code used for creating 'Field delineation' and 'Evapotranspiration' products for the 
EOAgriTwin project.
The different scripts and functions are divided into the following subfolders:

## evapTools
Workflow for evapotranspiration product

## fieldTools
Workflow for field delineation product

## FuncBox
collection of all functions needed to run evapTools and fieldTools. This folder contains several .py files that separate the functions thematically as well as the folder "other_repos". This folder contains copies(!) of repos from other users (pyDMS https://github.com/radosuav/pyDMS, pyTSEB https://github.com/hectornieto/pyTSEB, tfcl https://github.com/feevos/tfcl, S2LP_FORCE)

## yml
contains .yml files to create conda/mamba environments for workflow of evapTools
