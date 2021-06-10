#!/bin/bash
# The original pipeline script obtains the following results:

#Usage: ./training_mt.sh name_of_the_experiment

# Stages:
# 1. Data preparation 
# 2. Training

ARCH_MODEL_NAME="transformer_wmt_en_de" #['transformer_en_ru','transformer_wmt_en_de']
CUDA="0,1"
EXP_PATH="/home/cl/huyhien-v/Workspace/MT/experiments"
DATA_RAW="/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline"
PREFIX=$(date +'%Y-%m-%d')'-'
MAX_TOKEN="1000"
EXPID=$1

if [[ $1 == "" ]];
then
    EXPID=$((RANDOM%10000))
else
    EXPID=$1
fi


EXPID_PATH="$EXP_PATH/$PREFIX$EXPID"
if [[ -d "$EXPID_PATH" ]];
then 
    echo "exist"
else
    echo "create the experiments path... done"
    mkdir $EXPID_PATH
fi

DATA_BIN="$EXP_PATH/$PREFIX$EXPID/data-bin"
CHECKPOINT="$EXP_PATH/$PREFIX$EXPID/checkpoints"
echo "create the data-bin path... done"
if [[ -d "$DATA_BIN" ]];
then 
    echo "$DATA_BIN exists"
else
    echo "create the data-bin path... done"
    mkdir $DATA_BIN
fi
if [[ -d "$CHECKPOINT" ]];
then 
    echo "$CHECKPOINT exists"
else
    echo "create the checkpoint path... done"
    mkdir $CHECKPOINT
fi


echo "      Step 1: Processing data..."
# fairseq-preprocess --source-lang en --target-lang ru --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test 
#--destdir data-bin/voita.en-ru --thresholdtgt 0 --thresholdsrc 0 --workers 20

process_data_cm="fairseq-preprocess "
process_data_cm+="--source-lang en "
process_data_cm+="--target-lang ru "
process_data_cm+="--trainpref $DATA_RAW/train "
process_data_cm+="--validpref $DATA_RAW/valid "
process_data_cm+="--testpref $DATA_RAW/test "
process_data_cm+="--destdir $DATA_BIN/$EXPID "
process_data_cm+="--thresholdtgt 0 "
process_data_cm+="--workers 20 "

echo $process_data_cm
# $process_data_cm
echo "done."
echo "      Step 2: Training data..."

echo $training_cm
export CUDA_VISIBLE_DEVICES=$CUDA
$training_cm


# Note: Check space after each line of adding training_cm
training_cm=" "
# training_cm+="export CUDA_VISIBLE_DEVICES="$CUDA' '
training_cm+=" fairseq-train "$DATA_BIN/$EXPID' '
training_cm+="--arch $ARCH_MODEL_NAME "
training_cm+="--optimizer adam "
training_cm+="--adam-betas (0.9,0.98) "
training_cm+="--clip-norm 0.0 "
training_cm+="--lr 5e-4 "
training_cm+="--lr-scheduler inverse_sqrt "
training_cm+="--warmup-updates 4000 "
training_cm+="--dropout 0.1 "
training_cm+="--weight-decay 0.0001 "
training_cm+="--criterion label_smoothed_cross_entropy "
training_cm+="--label-smoothing 0.1  "
training_cm+="--max-tokens $MAX_TOKEN "
training_cm+="--keep-last-epochs 5 "
training_cm+="--eval-bleu "
training_cm+="--eval-bleu-args '{\"beam\":4,\"max_len_a\":1.2,\"max_len_b\":10}' "
training_cm+="--eval-bleu-detok moses "
training_cm+="--eval-bleu-remove-bpe "
training_cm+="--best-checkpoint-metric bleu "
training_cm+="--maximize-best-checkpoint-metric "
training_cm+="--save-dir $CHECKPOINT/$EXPID "
training_cm+="--num-workers 4 "
# training_cm+="--model-parallel-size 2 "
# training_cm+=" "
echo $training_cm
export CUDA_VISIBLE_DEVICES=$CUDA
# $training_cm

# CUDA_VISIBLE_DEVICES=0,1 fairseq-train data-bin/voita.en-ru 
#--arch transformer_wmt_en_de --optimizer adam --adam-betas '(0.9, 0.98)' 
#--clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 
#--dropout 0.1 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy 
#--label-smoothing 0.1 --max-tokens 4000 --keep-last-epochs 10 
#--eval-bleu --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' 
#--eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu 
#--maximize-best-checkpoint-metric --save-dir checkpoints/voita.en-ru.new 
#--num-workers 4 --model-parallel-size 2



