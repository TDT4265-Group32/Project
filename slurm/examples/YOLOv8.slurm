#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --job-name=RUN_TDT4265_YOLOv8_EXAMPLE
#SBATCH --time=0-00:15:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ          # Asking for a GPU
#SBATCH --mem=16G                 # Asking for 16GB RAM
#SBATCH --nodes=1                 # Asking for 1 nodes
#SBATCH --output=results/output.txt      # Specifying 'stdout'
#SBATCH --error=results/errors.err        # Specifying 'stderr'

#SBATCH --mail-user=cl@stud.ntnu.no
#SBATCH --mail-type=BEGIN,END,FAIL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "Running from this directory: $SLURM_SUBMIT_DIR"
echo "Name of job: $SLURM_JOB_NAME"
echo "ID of job: $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"

module purge

# Running your python file
module load Anaconda3/2023.09-0
conda activate ~/conda_envs/tdt4265-group32/
conda list
pip list
python ../src/examples/YOLOv8.py
