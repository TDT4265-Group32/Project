#!/bin/sh
#SBATCH --account=ie-idi
#SBATCH --job-name=RUN_TDT4265_YOLOv8
#SBATCH --time=0-01:00:00         # format: D-HH:MM:SS

#SBATCH --partition=GPUQ                    # Asking for a GPU
#SBATCH --mem=64G                           # Asking for 32GB RAM
#SBATCH --nodes=1                           # Asking for 1 nodes
#SBATCH --output=slurm/results/yolov8_output-%x-%j-%N.txt         # Specifying 'stdout'
#SBATCH --error=slurm/results/yolov8_errors-%x-%j-%N.err          # Specifying 'stderr'

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
module load cuDNN/8.8.0.121-CUDA-12.0.0
conda activate ~/conda_envs/tdt4265-group32/
cp -r runs prev_runs
rm -rf runs
python src/YOLOv8/train.py --dataset_path datasets/NAPLab-LiDAR --train_config configs/YOLOv8/NAPLab-LiDAR/train.json
python src/YOLOv8/val.py --model_path runs/detect/train/weights/best.pt --val_config configs/YOLOv8/NAPLab-LiDAR/train.json
python src/YOLOv8/predict.py --model_path runs/detect/train/weights/best.pt --pred_config configs/YOLOv8/NAPLab-LiDAR/pred.json --create_video True
