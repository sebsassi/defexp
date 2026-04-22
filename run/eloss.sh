#!/bin/bash
#SBATCH --job-name=eloss
#SBATCH -c 1
#SBATCH -n 1
#SBATCH --mem-per-cpu=600
#SBATCH --hint=nomultithread

function load_modules()
{
    module load $@
    if [[ $? -eq 0 ]]; then
        echo "Modules $@ loaded successfully."
    else
        echo "Failed to load modules $@."
        exit 1
    fi
}

MATERIAL=$1
COUNT=$2
SEED=1337
shift 2

if [[ -z "$WORK" ]]; then
    echo "Environment variable WORK is not defined."
    exit 1
fi
if [[ ! -d "$WORK" ]]; then
    echo "$WORK is not a directory."
    exit 1
fi

if [[ -z $PROJ ]]; then
    echo "Environment variable PROJ is not defined."
    exit 1
fi
if [[ ! -d "$PROJ" ]]; then
    echo "$PROJ is not a directory."
    exit 1
fi

MD_WORK=$WORK/mdsim
MD_PROJ=$PROJ/mdsim

if [ ! -d "$MD_WORK/lammps_work" ]; then
    echo "Directory $MD_WORK/lammps_work does not exist."
    exit 1
fi

if [ ! -d "$MD_WORK/dump" ]; then
    echo "Directory $MD_WORK/dump does not exist."
    exit 1
fi

if [ ! -d "$MD_WORK/thermo" ]; then
    echo "Directory $MD_WORK/thermo does not exist."
    exit 1
fi

if [ ! -d "$MD_WORK/logs" ]; then
    echo "Directory $MD_WORK/logs does not exist."
    exit 1
fi

if [ ! -d "$MD_WORK/eloss/$MATERIAL" ]; then
    echo "Directory $MD_WORK/eloss/$MATERIAL does not exist."
    exit 1
fi

cd $MD_WORK

if [ $? -ne 0 ]; then
    echo "Could not change directory to $MD_WORK."
    exit 1
fi

module purge
if [[ $? -eq 0 ]]; then
    echo "Modules unloaded successfully."
else
    echo "Failed to unload modules."
    exit 1
fi

load_modules $(cat $MD_PROJ/module_deps.txt)

if [ ! -d "$MD_PROJ/venv" ]; then
    echo "Virtual environment does not exist."
    exit 1
fi

source "$MD_PROJ/venv/bin/activate"
if [ $? -ne 0 ]; then
    echo "Failed to source virtual environment."
    exit 1
fi

srun python "$MD_PROJ/defexp/scripts/eloss.py" "$MATERIAL" "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$SEED" "$COUNT" \
    --config-dir "$MD_PROJ/defexp/samples" --res-dir "$MD_WORK" --work-dir "$MD_WORK" $@

deactivate

exit 0
