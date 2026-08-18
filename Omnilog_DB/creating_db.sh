#!/bin/bash

# loading module and env
module load python 
source /home/elouanln/scratch/Sandbox/VENV/bin/activate

# Take as input the directory that contain all relevants files
in_dir=$1


# Modify the creating-updat_result-DB.py file to execute the analysis
sed -i 's|general_dir = .*|general_dir = "'$in_dir'"|' "/home/elouanln/projects/def-jcomte/elouanln/Sandbox/Code/Phenotyping/Omnilog/analysis_pipeline/Omnilog_DB/creating-updat_result-DB.py"

# Execute the modified creating-updat_result-DB.py
echo "Executing creating-updat_result-DB.py for $in_dir..."
python3 "/home/elouanln/projects/def-jcomte/elouanln/Sandbox/Code/Phenotyping/Omnilog/analysis_pipeline/Omnilog_DB/creating-updat_result-DB.py"
echo "Execution completed for $in_dir."

# Setting back the default state of the python script
sed -i 's|general_dir = .*|general_dir = ""|' "/home/elouanln/projects/def-jcomte/elouanln/Sandbox/Code/Phenotyping/Omnilog/analysis_pipeline/Omnilog_DB/creating-updat_result-DB.py"
