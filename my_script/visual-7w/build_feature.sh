source $(dirname $0)/all_hypparams.sh

IMAGE_DIR=./my_dataset/${DATASET}/raw/images

for SPLIT in 'train' 'valid' 'test'
do
    python my_cli/build_grid_feature.py \
        --image-dir ${IMAGE_DIR} \
        --split ${SPLIT} \
        --processed-dir ${DATA_DIR} \
        --batch-size 128
done