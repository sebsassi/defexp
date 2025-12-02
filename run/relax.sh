#!/bin/bash
#SBATCH -t 10:00
#SBATCH --mem-per-cpu=600
#SBATCH --error="/home/%u/err/relax-%j.err"
#SBATCH --output="/home/%u/out/relax-%j.out"

SIMULATION=$1
MATERIAL=$2

MD_WORKDIR=/wrk-kappa/users/$USER/mdsim
if [ ! -d "$MD_WORKDIR" ]; then
    mkdir  $MD_WORKDIR
    if [ $? -ne 0 ]; then
        echo "Could not create directory $MD_WORKDIR."
        exit 1
    fi
fi

cd $MD_WORKDIR

if [ $? -ne 0 ]; then
    echo "Could not change directory to $MD_WORKDIR."
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

srun python "$PROJ/mdsim/defexp/scripts/relax.py" "$MATERIAL" \
    --config-dir "$PROJ/mdsim/defexp/samples" --work-dir "$MD_WORKDIR" --res-dir "$MD_WORKDIR"

deactivate

exit 0
