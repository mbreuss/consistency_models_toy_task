#!/bin/bash

############ GENERAL ENV SETUP ############
echo New Environment Name:
read envname

echo Creating new conda environment $envname 
conda create -n $envname python=3.8 -y -q

eval "$(conda shell.bash hook)"
conda activate $envname

echo
echo Activating $envname
if [[ "$CONDA_DEFAULT_ENV" != "$envname" ]]
then
    echo Failed to activate conda environment.
    exit 1
fi


############ PYTHON ############
echo Install mamba
conda install mamba -c conda-forge -y -q


############ REQUIRED DEPENDENCIES (PYBULLET) ############
echo Installing dependencies...

# Activate Mujoco Py Env Variables
conda activate $envname

# Install other PIP Dependencies
pip install torch 
pip install einops
pip install -U scikit-learn
pip install tqdm
pip install matplotlib
pip install numpy

echo Done installing all necessary packages. Please follow the next steps mentioned on the readme

pip install -e .
exit 0
