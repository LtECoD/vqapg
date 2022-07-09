DATASET=visual-7w
ARCH=vqac

DATA_DIR=./my_dataset/${DATASET}/processed
SAVE_DIR=./my_ckpt/${DATASET}/${ARCH}

if [ ! "`ls ${SAVE_DIR}`" = "" ]; then
    rm ${SAVE_DIR}/*
fi

fairseq-train ${DATA_DIR} \
    --user-dir my_eval/consistency_model \
    --save-dir ${SAVE_DIR} \
    --num-workers 0 \
    --no-epoch-checkpoints \
    \
    --optimizer adam \
    --lr 1e-3 \
    --clip-norm 5. \
    --batch-size 64 \
    --max-epoch 60 \
    --lr-scheduler reduce_lr_on_plateau \
    --lr-shrink 0.5 \
    --lr-patience 5 \
    \
    --task vqac_consistency \
    \
    --arch ${ARCH} \
    --dropout 0.5 \
    --embed-dim 300 \
    --token-embed-path /home/yangsen/dataset/glove.6B/glove.6B.300d.txt \
    \
    --criterion ${ARCH} \
