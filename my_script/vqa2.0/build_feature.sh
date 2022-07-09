source $(dirname $0)/all_hypparams.sh

TRAIN_IMAGE_DIR=./my_dataset/${DATASET}/raw/train2014
VAL_IMAGE_DIR=./my_dataset/${DATASET}/raw/val2014


python my_cli/build_grid_feature.py \
    --image-dir ${TRAIN_IMAGE_DIR} \
    --split train \
    --processed-dir ${DATA_DIR} \
    --batch-size 128

python my_cli/build_grid_feature.py \
    --image-dir ${TRAIN_IMAGE_DIR} \
    --split valid \
    --processed-dir ${DATA_DIR} \
    --batch-size 128

python my_cli/build_grid_feature.py \
    --image-dir ${VAL_IMAGE_DIR} \
    --split test \
    --processed-dir ${DATA_DIR} \
    --batch-size 128