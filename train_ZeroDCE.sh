#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Trains the ZeroDCE++ Model on own data
#
#  Script takes the following parameters:
#   -- Data Parameters --
#     [FRAMES] - Path to Frames Directory
#     [VALID]  - Ratio of Validation Frames to hold out
#     [RANDOM] - Random Seed
#   -- Model Parameters --
#     [PRETRAIN] - Pre-Trained Model to load from
#     [SCALING]  - Scaling Factor
#   -- Training Parameters --
#     [BATCH]  - Batch Size
#     [RATE]   - Learning Rate
#     [EPOCHS] - Max Epochs

#
#  USAGE:
#    This is designed to be run on the RTX90 GPUs, from the ZeroDCE directory:
#    srun --time=1-23:00:00 --gres=gpu:1 --nodelist=charles05 train_ZeroDCE.sh Frames_Raw_Ext 0.3 101 ZeroDCE++.pth 12 8 0.00001 20 &> logs/train_zdce_1e-5_20.log

#  Data Structures
#    This does not copy the Frames themselves: instead, they are specified at the path, relative to
#    /disk/scratch/${USER}/data/behaviour/ .
#    The Pre-Trained Model is loaded once from the relative path ~/models/ZeroDCE/Base/
#    Similarly, output is done directly to the home directory ~/models/ZeroDCE/Trained

####  Some Configurations
# Get and store the main Parameters
FRAMES_DIR=${1}
VALID_RATE=${2}
RAND_SEED=${3}

PRETRAIN=${4}
SCALING=${5}

BATCH_SIZE=${6}
LR=${7}
MAX_EPOCHS=${8}

# Derivative Values
OUT_NAME=B${BATCH_SIZE}_L${LR}_R${RAND_SEED}_S${SCALING}

# Path Values
DATA_DIR=/disk/scratch/${USER}/data/behaviour/${FRAMES_DIR}
OUT_DIR="${HOME}/models/ZeroDCE/Trained/${OUT_NAME}/"

# ===================
# Environment setup
# ===================
echo "Setting up Conda enviroment on ${SLURM_JOB_NODELIST}: Config=${OUT_NAME}"
echo "Using configuration: ${OUT_NAME}, with ${FRAMES_DIR} images and ${PATH_OFFSET} data split."
set -e # Make script bail out after first error
source activate py3rtx   # Activate Conda Environment
echo "Libraries from: ${LD_LIBRARY_PATH}"

# Setup NCCL Debug Status
export NCCL_DEBUG=INFO

# ===========
# Train Model
# ===========
echo " ===================================="
echo " Training Model (${OUT_NAME})"
mkdir -p "${OUT_DIR}"
python Zero-DCE++/lowlight_train.py \
  --images ${DATA_DIR} --validation_ratio ${VALID_RATE} --random_seed ${RAND_SEED} \
  --pretrain_model ${PRETRAIN} --scale_factor ${SCALING} --snapshots_folder ${OUT_DIR} \
  --batch_size ${BATCH_SIZE} --lr ${LR} --num_epochs ${MAX_EPOCHS}
echo "   == Training Done =="
echo ""

# ===========
# Nothing to copy, since I save directly to output disk
# ===========
echo "   ++ ALL DONE! Hurray! ++"
conda deactivate