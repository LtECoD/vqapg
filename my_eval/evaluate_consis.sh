
DATASET=$1
ARCH=$2
QAORDER=$3

GOLD_DIR=./my_dataset/${DATASET}/processed
if [ -z "${QAORDER}" ]; then
    PRED_ROOT_DIR=./my_result/${DATASET}/${ARCH}
else
    PRED_ROOT_DIR=./my_result/${DATASET}/${ARCH}/${QAORDER}
fi
MODEL_DIR=./my_ckpt/${DATASET}/vqac

function evaluate_consis(){
    PRED_DIR=${PRED_ROOT_DIR}/Epoch${1}

    python my_eval/metric_consis.py \
        --question-file ${PRED_DIR}/question.txt \
        --answer-file   ${PRED_DIR}/answer.txt \
        --imageid-file  ${GOLD_DIR}/test.image \
        --feature-dir   ${GOLD_DIR}/test-features-grid \
        --ckpt          ${MODEL_DIR}/checkpoint_best.pt \
        --output-file ${2}
}

if [ -z "${QAORDER}" ]; then
    FN=${ARCH}.consistency.txt
else
    FN=${ARCH}.${QAORDER}.consistency.txt
fi

echo > ./evaluation_result/${DATASET}/${FN}
echo 'last: ' >> ./evaluation_result/${DATASET}/${FN}
evaluate_consis _last ./evaluation_result/${DATASET}/${FN}

