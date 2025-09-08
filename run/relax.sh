#!/bin/bash
#SBATCH -M kale
#SBATCH -t 04:00
#SBATCH --mem-per-cpu=60
#SBATCH --error="relax-%j.err"
#SBATCH --output="relax-%j.out"

MY_WORKDIR=/wrk-kappa/users/$USER
cd $MY_WORKDIR

if [ $? -ne 0 ]
then
    echo "Couldn't change directory to" $MY_WORKDIR
    exit 1
fi

cp $HOME/script/* .
if [ $? -ne 0 ]
then
    echo "Failed to copy files to" `pwd`
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
    mkdir workEnv
    virtualenv workEnv
fi

if [ ! -d $PROJ/thresholds ]
then
    mkdir $PROJ/thresholds
fi

source workEnv/bin/activate

srun python relax.py $1

deactivate

exit 0
