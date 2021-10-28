#!/bin/bash
# Run contrastive test from the begining with a dictionary

CUDA_VISIBLE_DEVICES="0" #"CUDA_VISIBLE_DEVICES=0,1"

# Checkpoint to evaluate
checkpoint='/home/cl/huyhien-v/Workspace/MT/experiments/baseline1_5_full_split_400k/checkpoints/baseline1_5_full_para_split_400k/checkpoint_last.pt'

# Task to evaluate, default is `translation` or `document-translation`
task='document-translation'
batch='1'
# Criterion can be changed if we implement other loss function
criterion='label_smoothed_cross_entropy'
# echo "$task $criterion"
# exit
# output
destination_path='/home/is/huyhien-v/Dev/Temp/test'

# Source and Target language
src='en'
tgt='ru'

# BPE rules
src_bpe='/home/is/huyhien-v/Data/Bin/MT/en-ru/bpe/32k_1_5M/bpe.context_aware_1_5m.rules.en'
tgt_bpe='/home/is/huyhien-v/Data/Bin/MT/en-ru/bpe/32k_1_5M/bpe.context_aware_1_5m.rules.ru'

# Vocab
src_dict='/home/cl/huyhien-v/Workspace/MT/experiments/baseline1_5_full_split_400k/data-bin/baseline1_5m_full_split_400k00/dict.en.txt'
tgt_dict='/home/cl/huyhien-v/Workspace/MT/experiments/baseline1_5_full_split_400k/data-bin/baseline1_5m_full_split_400k00/dict.ru.txt'
# checkpoint='/home/is/huyhien-v/Dev/Trials/debug_envi/checkpoints/debug_envi/checkpoint_last.pt'


# checkpoint='/home/cl/huyhien-v/Workspace/MT/Test/modify-checkpoint/checkpoint465.pt'



count_loss_script='/home/is/huyhien-v/Dev/MT/fairseq-alter/supporting_scripts/count_loss_dataset.py'
voita_et_al_19_repo='/home/cl/huyhien-v/Workspace/MT/Voita_19/good-translation-wrong-in-context'
consistency_test_directory=$voita_et_al_19_repo"/consistency_testsets/scoring_data"
# consistency_test_directory='/home/is/huyhien-v/Data/Raw/MT/En-Ru/Contrastive/consistency_testsets/scoring_data'

random_seed="1234"
batch_size='512'
if [ -d "$destination_path" ];
then
    :
else
    mkdir $destination_path
fi

if [ -d "$destination_path/raw" ];
then
    :
else
    mkdir "$destination_path/raw"
fi

# data file

# deixis_dev_en="$consistency_test_directory/deixis_dev"
# deixis_dev_ru="$consistency_test_directory/deixis_dev.dst"
# deixis_test_en="$consistency_test_directory/deixis_test"
# deixis_test_ru="$consistency_test_directory/deixis_test.dst"
# ellipsis_infl_en="$consistency_test_directory/ellipsis_infl"
# ellipsis_infl_ru="$consistency_test_directory/ellipsis_infl.dst"
# ellipsis_vp_en="$consistency_test_directory/ellipsis_vp"
# ellipsis_vp_ru="$consistency_test_directory/ellipsis_vp.dst"
# lex_cohesion_dev_en="$consistency_test_directory/lex_cohesion_dev"
# lex_cohesion_dev_ru="$consistency_test_directory/lex_cohesion_dev.dst"
# lex_cohesion_test_en="$consistency_test_directory/lex_cohesion_test"
# lex_cohesion_test_ru="$consistency_test_directory/lex_cohesion_test.dst"

# all_tests=('deixis_dev' 'deixis_test' 'ellipsis_infl' 'ellipsis_vp' 'lex_cohesion_dev' 'lex_cohesion_test');
all_tests=('deixis_test' 'ellipsis_infl' 'ellipsis_vp' 'lex_cohesion_test');


if [ "$1" = "new" ];
then
    # create binary dictionary
    for test in "${all_tests[@]}"; do
        echo "Processing $test ..."
        # echo "    copy..."
        # cp $consistency_test_directory"/$test.src" $destination_path"/raw/$test.$src"
        # cp $consistency_test_directory"/$test.dst" $destination_path"/raw/$test.$tgt"
        
        echo "    lowercase..."
        awk '{print tolower($0)}' < $consistency_test_directory"/$test.src" > $destination_path"/raw/$test.lower.$src"
        awk '{print tolower($0)}' < $consistency_test_directory"/$test.dst" > $destination_path"/raw/$test.lower.$tgt"
        
        echo "    apply bpe..."
        subword-nmt apply-bpe -c $src_bpe < $destination_path"/raw/$test.lower.$src" > $destination_path"/raw/$test.$src"
        subword-nmt apply-bpe -c $tgt_bpe < $destination_path"/raw/$test.lower.$tgt" > $destination_path"/raw/$test.$tgt"
        
        echo "    create dataset..."
        echo "        fairseq-preprocess --source-lang $src --target-lang $tgt --testpref $destination_path/raw/$test --thresholdtgt 0 --thresholdsrc 0 --workers 20 --srcdict $src_dict --tgtdict $tgt_dict --destdir $destination_path/$test --seed $random_seed"
        rm -rf $destination_path"/raw/$test.lower.$src"
        rm -rf $destination_path"/raw/$test.lower.$tgt"

        fairseq-preprocess --source-lang $src --target-lang $tgt \
            --testpref "$destination_path/raw/$test" \
            --thresholdtgt 0 --thresholdsrc 0 --workers 20 \
            --srcdict $src_dict \
            --tgtdict $tgt_dict \
            --destdir "$destination_path/$test"  --seed $random_seed
        # break
    done
fi

for test in "${all_tests[@]}"; do
        echo "Validating $test ..."
        # echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES fairseq-generate $destination_path/$test --path $checkpoint --results-path $destination_path/$test"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $count_loss_script $destination_path/$test --task $task --path $checkpoint --results-path $destination_path/$test --batch-size 1 --valid-subset test --criterion $criterion "
        # Note that batch-size is always 1
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $count_loss_script $destination_path/$test --task $task --path $checkpoint --results-path $destination_path/$test --batch-size $batch --valid-subset 'test' --criterion $criterion 
        #output file is alwas `$destination_path/$test/generate-loss-test.txt`        
        echo "    Scoring..."
        python $voita_et_al_19_repo/scripts/evaluate_consistency.py \
            --repo-dir $voita_et_al_19_repo \
            --test $test \
            --scores $destination_path/$test/generate-loss-test.txt > $destination_path/$test/$test".consistency.txt"
done

echo "Done"