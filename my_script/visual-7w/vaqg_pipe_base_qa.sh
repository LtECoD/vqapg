source $(dirname $0)/all_hypparams.sh

ARCH=vaqg_pipe_base
QA_ORDER=qa

if [ ${QA_ORDER} = aq ]; then
    FIRST="answer"
    SECOND="question"
else
    FIRST="question"
    SECOND="answer"
fi

DATA_DIR=./my_dataset/${DATASET}/processed
SAVE_DIR=./my_ckpt/${DATASET}/${ARCH}/${QA_ORDER}/${FIRST}

# Train First
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
    --task vaqg_pipe \
    --target ${FIRST} \
    \
    --arch ${ARCH} \
    --dropout ${DROPOUT} \
    --token-embed-path ${TOKENEMBEDPATH} \
    --embed-dim ${EMBEDDIM} \
    \
    --criterion ${ARCH} \


# Generate First
PRED_ROOT_DIR=./my_result/${DATASET}/${ARCH}/${QA_ORDER}

function generate_first(){
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
        --task vaqg_pipe \
        --target ${FIRST} \

    EVAL_FILE=${PRED_DIR}/generate-test.txt
    TARGET_FILE=${PRED_DIR}/${FIRST}.txt

    grep ^H ${EVAL_FILE} | awk -F "-| " '{print $2""$0}' | sort -n | cut -f4- > ${TARGET_FILE}
    mv ${EVAL_FILE} ${PRED_DIR}/${FIRST}.generate-test.txt
}

generate_first "_last"


# Train Second
SAVE_DIR=./my_ckpt/${DATASET}/${ARCH}/${QA_ORDER}/${SECOND}
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
    --task vaqg_pipe \
    --target ${SECOND} \
    --preliminary ${FIRST} \
    \
    --arch ${ARCH} \
    --dropout ${DROPOUT} \
    --token-embed-path ${TOKENEMBEDPATH} \
    --embed-dim ${EMBEDDIM} \
    \
    --criterion ${ARCH} \

# Generate Second
PRED_ROOT_DIR=./my_result/${DATASET}/${ARCH}/${QA_ORDER}

function generate_second(){
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
        --task vaqg_pipe \
        --target ${SECOND} \
        --preliminary ${FIRST} \
        --preliminary-path ${PRED_DIR}/${FIRST}.txt \

    EVAL_FILE=${PRED_DIR}/generate-test.txt
    TARGET_FILE=${PRED_DIR}/${SECOND}.txt

    grep ^H ${EVAL_FILE} | awk -F "-| " '{print $2""$0}' | sort -n | cut -f4- > ${TARGET_FILE}
    mv ${EVAL_FILE} ${PRED_DIR}/${SECOND}.generate-test.txt
}

generate_second "_last"