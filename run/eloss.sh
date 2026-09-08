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

MD_PROJ=$PROJ/mdsim
MD_WORK=$WORK/mdsim

if [ ! -d "$MD_PROJ" ]; then
    echo "Directory $MD_PROJ does not exist."
    exit 1
fi

if [ ! -d "$MD_WORK" ]; then
    echo "Directory $MD_WORK does not exist."
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

srun python "$MD_PROJ/defexp/scripts/eloss.py" -j "$SLURM_JOB_ID" -p "$SLURM_ARRAY_TASK_ID" \
    --config-dir "$MD_PROJ/defexp/samples" --res-dir "$MD_WORK" --work-dir "$MD_WORK" -n "$SLURM_CPUS_PER_TASK" $@

deactivate

exit 0
