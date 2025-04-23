# Niccolo Forte's PhD Repository

#### Optional but recommended:
Please run the following terminal commands to ensure use of a Vitrual Environment (venv) and local installation of all required packages in the `requirements.txt` scripts upon download of this repo:

Access repo directory: `cd path\to\repo\p1git-Lattices`  
Initiate venv: `pytjon -m venv venv`
Activate venv: `venv\Scripts\activate`
Install required packages: `pip install -r requirements.txt`


## ABAQUS Simulation Files
ABAQUS scipts for local and HPC simulation execution are found in the `SIMscripts` directory.

Files beginning with "A" are the core python 3 scripts to run ABAQUS/Explicit jobs (A1/A-HPC-1) and their post processing (A2/A-HPC-2). A3 is an alternative post processing targeting the output sctructure required for the developmenent of continuum plots.  
Files beginning with "B" are the main BASH (`.sh`) files used to submit jobs to HPC, transfer files between local and/or remote directories, and generally handle simulations files after completion. Custom scripts are present of a variety of different useful tasks.  
HPC job scripts are exact copies of local job scripts adapted for HPC runs.The main differences include  of path specification and some variable definitions being provided from the job's BASH file. 

## Data Processing
Jupyter notebooks (`.ipynb`) used for data processing and visualizing results are stored in the `code` directory. Within this directory, the `resources` directory stores PYTHON (`.py`) scripts with general, versatile functions that are imported into a variety of notebooks.

## Machine Learning
Machine learning Jupyter notebooks are also found in the `code` directory. These include scripts to process and format data for machine learning scripts, and a variety of models. For each model, the script name represent the model within.