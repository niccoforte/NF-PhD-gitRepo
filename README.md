# Niccolo Forte

## ABAQUS Simulation Files
ABAQUS and HPC simulation files are found in the `SIMscripts` directory.

Files beginning with "A" are the core python scripts to run ABAQUS/Explicit jobs (A1) and their post processing (A2). 
Files beginning with "B" are the main BASH (`.sh`) files used to submit jobs to HPC.
Scripts used to submit HPC jobs are identical to those used to submit local jobs, with the exception of path specification and some variable definitions being provided from the job's BASH file. 

## Data Processing
Jupyter notebooks (`.ipynb`) used for data processing and visualizing results are stored in the `code` directory. Within this directory, the `resources` directory stores PYTHON (`.py`) scripts with general, versatile functions that are imported into a variety of notebooks.

## Machine Learning
Machine learning Jupyter notebooks are also found in the `code` directory. These include scripts to process and format data for machine learning scripts, and a variety of models. For each model, the script name represent the model within.