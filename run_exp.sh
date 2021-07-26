# Run experiments from A-Z
# Note that your data have to be lowercased and be bpelized before running this script
# You need to specify
#   EXP_PATH: path of the experiments. This script will copy data, split and train
#   DATA_RAW: path of the raw dat: train.src, train.tgt, test.src, test.tgt, valid.src, valid.tgt
#       For example: train.en, train.ru, test.en, test.ru, valid.en, valid.ru
#   CUDA: number of GPU. Ex: 0,1
#   src_lang: source language. Ex: en
#   tgt_lang: target langauge. Ex: ru
# Copy data and split to small parts
# Input:
#   $1: number of steps
#       1: preparing data + training
#       2: preparing data
#       3: training
#

# EXP_PATH="/home/cl/huyhien-v/Workspace/MT/experiments/baseline1_5_original"
EXP_PATH="/home/cl/huyhien-v/Workspace/MT/experiments/debug"
# DATA_RAW="/home/cl/huyhien-v/Workspace/MT/data/voita_19/baseline1.5/separated_with_bpe"
# DATA_RAW="/home/cl/huyhien-v/Workspace/MT/data/voita_19/baseline1.5/original_with_bpe"
DATA_RAW="/home/cl/huyhien-v/Workspace/MT/data/voita_19/baseline1.5/debug"
FAIRSEQ="/home/cl/huyhien-v/Workspace/MT/my_fairseq/fairseq"
prefix='voita-en-ru-1_5m' #'multi_att_en-ru'

WANDB_PROJECT='voita-en-ru-1_5m'
SEED='11234'
thresholdtgt="0"
thresholdsrc="0"
workers="20"

CUDA="0"
ARCH_MODEL_NAME="transformer_wmt_en_de" #['transformer_en_ru','transformer_wmt_en_de']
MAX_TOKEN="1000"



NUMBER_OF_LINES_IN_FILE="100000" #old value: 200000
EXPID=$1
DATA_BIN="$EXP_PATH/data-bin"
CHECKPOINT="$EXP_PATH/checkpoints"
WANDB="$EXP_PATH/wandb"
SPLITTED_FOLDER='train_splitted'
src_lang='en'
tgt_lang='ru'
TEMP_DIR="$EXP_PATH/temp"
# echo $1
if [ "$1" = "1" ] || [ "$1" = "2" ] ; then
    echo "Preparing data..."
    echo "   Creating the data-bin folder..."
    if [ -d "$DATA_BIN" ];
    then
        echo "$DATA_BIN exists"
    else
        echo "      create the data-bin folder... done"
        mkdir $DATA_BIN
    fi
    echo "   Creating the checkpoints folder..."
    if [ -d "$CHECKPOINT" ];
    then
        echo "$CHECKPOINT exists"
    else
        echo "      create the checkpoints folder... done"
        mkdir $CHECKPOINT
    fi
    echo "   Creating the wandb folder..."
    if [ -d "$WANDB" ];
    then
        echo "$WANDB exists"
    else
        echo "      create the wandb folder... done"
        mkdir $WANDB
    fi
    echo "   Creating the temp folder..."
    if [ -d "$TEMP_DIR" ];
    then
        echo "$TEMP_DIR exists"
    else
        echo "      create the temp folder... done"
        mkdir $TEMP_DIR
    fi
    echo "   Creating small parts of the big source files"
    # argument of python file:
    #   path_to_source_of_data    source_language    target_language    experiment_path    number_of_line_in_one_file
    python $FAIRSEQ/supporting_scripts/split_big_file_to_data_bin.py \
        $DATA_RAW    $src_lang    $tgt_lang    $EXP_PATH    \
        $NUMBER_OF_LINES_IN_FILE   $SPLITTED_FOLDER
    echo "   Copying test and valid files into folders"
    python $FAIRSEQ/supporting_scripts/copy_valid_test_to_experiment_folder.py \
        $DATA_RAW    $src_lang    $tgt_lang  $EXP_PATH"/"$SPLITTED_FOLDER
    echo "   Running process data for full set and create data"
    python $FAIRSEQ/supporting_scripts/run_data_processing.py \
        $DATA_RAW    $src_lang    $tgt_lang  $TEMP_DIR\
        $thresholdsrc    $thresholdtgt    $workers\
        $EXP_PATH   $EXP_PATH"/"$SPLITTED_FOLDER  $prefix
fi
training_cm=''
if [ "$1" = "1" ] || [ "$1" = "3" ] ; then
    echo "Training data ..."
    # python $FAIRSEQ/supporting_scripts/run_training.py \
    #     $ARCH_MODEL_NAME
    SUB_DATA=''
    all_folders=$(ls $DATA_BIN);
    for folder in $all_folders;
    do
        SUB_DATA="$SUB_DATA$DATA_BIN/$folder:"
    done
    SUB_DATA=`echo $SUB_DATA|sed 's/.$//'`

    training_cm=""
    training_cm="$training_cm fairseq-train $SUB_DATA"
    training_cm="$training_cm --arch $ARCH_MODEL_NAME "
    training_cm="$training_cm --optimizer adam "
    training_cm="$training_cm --adam-betas '(0.9,0.98)' "
    training_cm="$training_cm --clip-norm 0.0 "
    training_cm="$training_cm --lr 5e-4 "
    training_cm="$training_cm --lr-scheduler inverse_sqrt "
    training_cm="$training_cm --warmup-updates 4000 "
    training_cm="$training_cm --dropout 0.1 "
    training_cm="$training_cm --criterion label_smoothed_cross_entropy "
    training_cm="$training_cm --label-smoothing 0.1 "
    training_cm="$training_cm --max-tokens $MAX_TOKEN "
    training_cm="$training_cm --keep-last-epochs 5 "
    training_cm="$training_cm --eval-bleu "
    training_cm="$training_cm --eval-bleu-args '{\"beam\":4,\"max_len_a\":1.2,\"max_len_b\":10}' "
    training_cm="$training_cm --eval-bleu-detok moses "
    training_cm="$training_cm --eval-bleu-remove-bpe "
    training_cm="$training_cm --best-checkpoint-metric bleu "
    training_cm="$training_cm --maximize-best-checkpoint-metric "
    training_cm="$training_cm --save-dir $CHECKPOINT/$prefix "
    training_cm="$training_cm --num-workers 4 "
    training_cm="$training_cm --update-freq 8 "
    training_cm="$training_cm --wandb-project $WANDB_PROJECT --seed $SEED"
    # training_cm+=" --max-tokens $MAX_TOKEN "
    # training_cm+=" --keep-last-epochs 5 "
    # training_cm+=" --eval-bleu "
    # training_cm+=" --eval-bleu-args '{\"beam\":4,\"max_len_a\":1.2,\"max_len_b\":10}' "
    # training_cm+=" --eval-bleu-detok moses "
    # training_cm+=" --eval-bleu-remove-bpe "
    # training_cm+=" --best-checkpoint-metric bleu "
    # training_cm+=" --maximize-best-checkpoint-metric "
    # training_cm+=" --save-dir $CHECKPOINT/$EXPID "
    # training_cm+=" --num-workers 4 "
    echo $training_cm
    # export CUDA_VISIBLE_DEVICES=$CUDA
    # $training_cm
fi




