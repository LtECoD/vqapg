source $(dirname $0)/all_hypparams.sh

python my_cli/build_vocab.py \
    --trainpref ${DATA_DIR}/train \
    --destdir ${DATA_DIR} \
    --question question \
    --answer answer \
    --target target \
    --joined-dictionary

