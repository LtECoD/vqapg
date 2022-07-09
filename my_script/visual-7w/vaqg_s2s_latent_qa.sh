source $(dirname $0)/all_hypparams.sh

# Train
ARCH=vaqg_s2s_latent
QA_ORDER=qa
DATA_DIR=./my_dataset/${DATASET}/processed
SAVE_DIR=./my_ckpt/${DATASET}/${ARCH}/${QA_ORDER}

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
    --task vaqg_s2s \
    --qa-order ${QA_ORDER} \
    \
    --arch ${ARCH} \
    --dropout ${DROPOUT} \
    --token-embed-path ${TOKENEMBEDPATH} \
    --embed-dim ${EMBEDDIM} \
    \
    --criterion ${ARCH} \
    --free-bits-p ${FREEBITSP}

# Generate
PRED_ROOT_DIR=./my_result/${DATASET}/${ARCH}/${QA_ORDER}

if [ ! "`ls ${PRED_ROOT_DIR}`" = "" ]; then
    rm -r ${PRED_ROOT_DIR}/*
fi

function generate(){
    MODEL_FILE=${SAVE_DIR}/checkpoint${1}.pt
    PRED_DIR=${PRED_ROOT_DIR}/Epoch${1}

    python my_cli/vaqg_generate.py ${DATA_DIR} \
        --user-dir ${USERDIR} \
        \
        --path ${MODEL_FILE} \
        --beam ${BEAM} \
        --results-path ${PRED_DIR} \
        --batch-size ${INFERENCEBATCHSIZE} \
        \
        --task vaqg_s2s \
        --qa-order ${QA_ORDER} \

    EVAL_FILE=${PRED_DIR}/generate-test.txt
    HYPO_FILE=${PRED_DIR}/hypothesis.txt
    ANSWER_FILE=${PRED_DIR}/answer.txt
    QUESTION_FILE=${PRED_DIR}/question.txt

    grep ^H ${EVAL_FILE} | awk -F "-| " '{print $2""$0}' | sort -n | cut -f4- > ${HYPO_FILE}
    awk -F ' \[sep\] ' '{print $1}' ${HYPO_FILE} > ${PRED_DIR}/question.txt
    awk -F ' \[sep\] ' '{print $2}' ${HYPO_FILE} > ${PRED_DIR}/answer.txt
}

generate "_best"
generate "_last"
