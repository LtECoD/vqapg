source $(dirname $0)/all_hypparams.sh

# Train
ARCH=vaqg_s2s_latent_scale
QA_ORDER=aq
DATA_DIR=./my_dataset/${DATASET}/interactive
SAVE_DIR=./my_ckpt/${DATASET}/${ARCH}/${QA_ORDER}


# Generate
PRED_ROOT_DIR=./my_result/${DATASET}/interactive/${ARCH}/${QA_ORDER}


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

}

generate "_last"