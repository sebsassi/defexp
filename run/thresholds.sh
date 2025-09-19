#!/bin/bash
#SBATCH --job-name=thresholds
#SBATCH -M kale
#SBATCH -c 1
#SBATCH -n 1
#SBATCH -t 23:00:00
#SBATCH --mem-per-cpu=60
#SBATCH --error="err/thresholds-%j.err"
#SBATCH --output="out/thresholds-%j.out"
#SBATCH --hint=nomultithread

MY_WORKDIR=/wrk-kappa/users/$USER
cd $MY_WORKDIR

srun lscpu

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

srun python thresholds.py $1 $SLURM_JOB_ID $SLURM_ARRAY_TASK_ID $2 $3

deactivate

exit 0
