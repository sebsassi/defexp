#!/bin/bash
#SBATCH --job-name=eloss
#SBATCH -M kale
#SBATCH -c 1
#SBATCH -n 1
#SBATCH -t 23:00:00
#SBATCH --mem-per-cpu=600
#SBATCH --error="err/eloss-%j.err"
#SBATCH --output="out/eloss-%j.out"
#SBATCH --hint=nomultithread

MATERIAL=$1
COUNT=$2
shift 2

srun lscpu

MD_WORKDIR=/wrk-kappa/users/$USER/mdsim

if [ ! -d "$MD_WORKDIR/lampps_work" ]; then
    echo "Directory $MD_WORKDIR/lammps_work does not exist."
    exit 1
fi

if [ ! -d "$MD_WORKDIR/dump" ]; then
    echo "Directory $MD_WORKDIR/dump does not exist."
    exit 1
fi

if [ ! -d "$MD_WORKDIR/thermo" ]; then
    echo "Directory $MD_WORKDIR/thermo does not exist."
    exit 1
fi

if [ ! -d "$MD_WORKDIR/logs" ]; then
    echo "Directory $MD_WORKDIR/logs does not exist."
    exit 1
fi

if [ ! -d "$MD_WORKDIR/eloss/$MATERIAL" ]; then
    echo "Directory $MD_WORKDIR/eloss/$MATERIAL does not exist."
    exit 1
fi

cd $MD_WORKDIR

if [ $? -ne 0 ]; then
    echo "Couldn't change directory to" $MY_WORKDIR
    exit 1
fi

module purge
if [ $? -eq 0 ]; then
    echo "Modules unloaded successfully."
else
    echo "Failed to unload modules."
    exit 1
fi

module load FFTW
if [ $? -eq 0 ]; then
    echo "Module FFTW loaded successfully."
else
    echo "Failed to load FFTW."
    exit 1
fi

module load Python
if [ $? -eq 0 ]; then
    echo "Module Python loaded successfully."
else
    echo "Failed to load Python."
    exit 1
fi

if [ ! -d "$PROJ/mdsim/venv" ]; then
    echo "Virtual environment does not exist."
    exit 1
fi

source "$PROJ/mdsim/venv/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to source virtual environment."
    exit 1
fi

srun python "$PROJ/defexp/scripts/eloss.py" "$MATERIAL" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$COUNT" \
    --config-dir "$PROJ/mdsim/defexp/samples" --res-dir "$MD_WORKDIR" --work-dir "$MD_WORKDIR" 

deactivate

exit 0
