
DATASET=$1
ARCH=$2
QAORDER=$3

GOLD_DIR=./my_dataset/${DATASET}/processed
if [ -z "${QAORDER}" ]; then
    PRED_ROOT_DIR=./my_result/${DATASET}/${ARCH}
else
    PRED_ROOT_DIR=./my_result/${DATASET}/${ARCH}/${QAORDER}
fi

function evaluate_ngram(){
    PRED_DIR=${PRED_ROOT_DIR}/Epoch${1}

    python my_eval/metric_ngram.py \
        --gold-answer ${GOLD_DIR}/test.answer \
        --gold-question ${GOLD_DIR}/test.question \
        --image-id ${GOLD_DIR}/test.image \
        --pred-answer ${PRED_DIR}/answer.txt \
        --pred-question ${PRED_DIR}/question.txt \
        --output-file ${2}
}

if [ -z "${QAORDER}" ]; then
    FN=${ARCH}.ngram.txt
else
    FN=${ARCH}.${QAORDER}.ngram.txt
fi

# echo best: > ./evaluation_result/${DATASET}/${FN}
# evaluate_ngram _best ./evaluation_result/${DATASET}/${FN}

echo > ./evaluation_result/${DATASET}/${FN}
echo last: >> ./evaluation_result/${DATASET}/${FN}
evaluate_ngram _last ./evaluation_result/${DATASET}/${FN}