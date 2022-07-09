source $(dirname $0)/all_hypparams.sh

# Train
ARCH=jvaqg_latent_scale
DATA_DIR=./my_dataset/${DATASET}/interactive
SAVE_DIR=./my_ckpt/${DATASET}/${ARCH}


# Generate
PRED_ROOT_DIR=./my_result/${DATASET}/interactive/${ARCH}


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
}

generate "_last"

    

