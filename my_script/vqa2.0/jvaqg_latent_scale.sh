source $(dirname $0)/all_hypparams.sh

# Train
ARCH=jvaqg_latent_scale
DATA_DIR=./my_dataset/${DATASET}/processed
SAVE_DIR=./my_ckpt/${DATASET}/${ARCH}

if [ ! "`ls ${SAVE_DIR}`" = "" ]; then
    rm ${SAVE_DIR}/*
fi

fairseq-train ${DATA_DIR} \
    --user-dir ${USERDIR} \
    --save-dir ${SAVE_DIR} \
    --num-workers ${NUMWORKERS} \
    --no-epoch-checkpoints \
    \
    --optimizer ${OPTIMIZER} \
    --lr ${LR} \
    --clip-norm ${CLIPNORM} \
    --batch-size ${BATCHSIZE} \
    --max-epoch ${MAXEPOCH} \
    --lr-scheduler ${LRSCHEDULER} \
    --lr-shrink ${LRSHRINK} \
    --lr-patience ${LRPATIENCE} \
    \
    --task jvaqg \
    --do-answer True \
    --do-question True \
    \
    --arch ${ARCH} \
    --dropout ${DROPOUT} \
    --share-token-embeddings ${SHARETOKENEMBEDDINGS} \
    --token-embed-path ${TOKENEMBEDPATH} \
    --embed-dim ${EMBEDDIM} \
    \
    --criterion ${ARCH} \
    --free-bits-p ${FREEBITSP} \
    --free-bits-b ${FREEBITSB} \
    --block-loss-weight ${BLOCKLOSSWEIGHT}


# Generate
PRED_ROOT_DIR=./my_result/${DATASET}/${ARCH}

if [ ! "`ls ${PRED_ROOT_DIR}`" = "" ]; then
    rm -r ${PRED_ROOT_DIR}/*
fi

function generate(){
    MODEL_FILE=${SAVE_DIR}/checkpoint${1}.pt
    PRED_DIR=${PRED_ROOT_DIR}/Epoch${1}

    python my_cli/jvaqg_generate.py ${DATA_DIR} \
        --user-dir my_module \
        \
        --path ${MODEL_FILE} \
        --beam ${BEAM} \
        --results-path ${PRED_DIR} \
        --batch-size ${INFERENCEBATCHSIZE} \
        \
        --task jvaqg \
        --do-answer True \
        --do-question True \

    EVAL_FILE=${PRED_DIR}/generate-test.txt
    ANSWER_OUT_FILE=${PRED_DIR}/answer.txt
    QUESTION_OUT_FILE=${PRED_DIR}/question.txt

    grep ^A ${EVAL_FILE} | awk -F "-| " '{print $2""$0}' | sort -n | cut -f4- > ${ANSWER_OUT_FILE}
    grep ^Q ${EVAL_FILE} | awk -F "-| " '{print $2""$0}' | sort -n | cut -f4- > ${QUESTION_OUT_FILE}
}

generate "_best"
generate "_last"

    

