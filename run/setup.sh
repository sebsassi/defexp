#!/bin/bash
#SBATCH -M kale
#SBATCH -t 04:00
#SBATCH --mem-per-cpu=60
#SBATCH --error="setup-%j.err"
#SBATCH --output="setup-%j.out"

MY_WORKDIR=/wrk-kappa/users/$USER
cd $MY_WORKDIR

if [ $? -ne 0 ]
then
    echo "Couldn't change directory to" $MY_WORKDIR
    exit 1
fi

cp $HOME/script/* $MY_WORKDIR
if [ $? -ne 0 ]
then
    echo "Failed to copy files to" $MY_WORKDIR
    exit 1
fi

module purge
if [ $? -eq 0 ]
then
    echo "Modules unloaded successfully."
else
    echo "Failed to unload moduels."
    exit 1
fi

module load FFTW
if [ $? -eq 0 ]
then
    echo "Module FFTW loaded successfully."
else
    echo "Failed to load FFTW."
    exit 1
fi

module load Python/3.8.6-GCCcore-10.2.0
if [ $? -eq 0 ]
then
    echo "Module Python loaded successfully."
else
    echo "Failed to load Python."
    exit 1
fi

if [ ! -d workEnv ]
then
    echo "Directory workEnv doesn't exist. Did you forget to run relax.sh?"
    exit 1
fi

if [ ! -d $PROJ/thresholds ]
then
    echo "Directory" $PROJ/thresholds "doesn't exist. Did you forget to run relax.sh?"
    exit 1
fi

source workEnv/bin/activate

pip install numpy
pip install ase

deactivate

exit 0
