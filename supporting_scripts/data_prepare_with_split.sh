#!/bin/bash

directory='/home/is/huyhien-v/Data/Raw/MT/En-Vi/uet'
destination='/home/is/huyhien-v/Data/Bin/debug_envi'
src_lang='en'
tgt_lang='vi'

# directory='/home/cl/huyhien-v/Workspace/MT/data/voita_19/context_agnostic'
# directory='/home/cl/huyhien-v/Workspace/MT/data/voita_19/context_aware'

# destination='/home/cl/huyhien-v/Workspace/MT/experiments/baseline1_5_last_sent_split'
# destination='/home/cl/huyhien-v/Workspace/MT/experiments/baseline6m_split_400k'
# destination='/home/cl/huyhien-v/Workspace/MT/experiments/baseline1_5_full_split_400k'

destination_raw=$destination'/raw'
destination_split=$destination'/raw/splits'
destination_data_raw=$destination'/raw/data-raw'
destination_dictionary=$destination'/raw/data-raw/dictionary'
number_of_line=100000
# src_lang='en'
# tgt_lang='ru'




output_src_tgt=$destination'/output_'$src_lang'-'$tgt_lang
mode=$1 # 'last'
# prefix='baseline1_5m_full_split_400k'
# prefix='baseline6m_split_400k'
prefix='debug_envi'
random_seed=1

# BPE_FOR_VALID_AND_TEST=$1

if [ -d "$destination" ];
then
    :
else
    mkdir $destination
fi

if [ -d "$destination_raw" ];
then
    :
else
    mkdir $destination_raw
fi

if [ -d "$output_src_tgt" ];
then
    :
else
    mkdir $output_src_tgt
fi

if [ -d "$destination_split" ];
then
    :
else
    mkdir $destination_split
fi

if [ -d "$destination_data_raw" ];
then
    :
else
    mkdir $destination_data_raw
fi

if [ -d "$destination_dictionary" ];
then
    :
else
    mkdir $destination_dictionary
fi



en_train=$directory'/'$src_lang'_train'
en_test=$directory'/'$src_lang'_test'
en_dev=$directory'/'$src_lang'_dev'
ru_train=$directory'/'$tgt_lang'_train'
ru_test=$directory'/'$tgt_lang'_test'
ru_dev=$directory'/'$tgt_lang'_dev'

# rename and lowercase
train_en=$destination_raw'/train.'$src_lang
test_en=$destination_raw'/test.'$src_lang
valid_en=$destination_raw'/valid.'$src_lang
train_ru=$destination_raw'/train.'$tgt_lang
test_ru=$destination_raw'/test.'$tgt_lang
valid_ru=$destination_raw'/valid.'$tgt_lang


if [ "$1" = "last" ];
then 
    echo "lowercase and select last sentence... "
    awk '{print tolower($0)}' < $en_train | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $destination_raw'/'$src_lang'_train.lower'
    awk '{print tolower($0)}' < $en_test | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $destination_raw'/'$src_lang'_test.lower'
    awk '{print tolower($0)}' < $en_dev | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $destination_raw'/'$src_lang'_dev.lower'

    awk '{print tolower($0)}' < $ru_train | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $destination_raw'/'$tgt_lang'_train.lower'
    awk '{print tolower($0)}' < $ru_test | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $destination_raw'/'$tgt_lang'_test.lower'
    awk '{print tolower($0)}' < $ru_dev | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $destination_raw'/'$tgt_lang'_dev.lower'
else
    echo "lowercase..."
    awk '{print tolower($0)}' < $en_train > $destination_raw'/'$src_lang'_train.lower'
    awk '{print tolower($0)}' < $en_test > $destination_raw'/'$src_lang'_test.lower'
    awk '{print tolower($0)}' < $en_dev > $destination_raw'/'$src_lang'_dev.lower'

    awk '{print tolower($0)}' < $ru_train > $destination_raw'/'$tgt_lang'_train.lower'
    awk '{print tolower($0)}' < $ru_test > $destination_raw'/'$tgt_lang'_test.lower'
    awk '{print tolower($0)}' < $ru_dev > $destination_raw'/'$tgt_lang'_dev.lower'
fi

echo "subword..."
subword-nmt learn-bpe -s 32000 < $destination_raw'/'$src_lang'_train.lower' > $destination_raw'/bpe.rules.'$src_lang
subword-nmt learn-bpe -s 32000 < $destination_raw'/'$tgt_lang'_train.lower' > $destination_raw'/bpe.rules.'$tgt_lang

echo "applying bpe..."
echo '  '$src_lang'...'
subword-nmt apply-bpe -c $destination_raw'/bpe.rules.'$src_lang < $destination_raw'/'$src_lang'_train.lower' > $train_en
subword-nmt apply-bpe -c $destination_raw'/bpe.rules.'$src_lang < $destination_raw'/'$src_lang'_test.lower' > $test_en
subword-nmt apply-bpe -c $destination_raw'/bpe.rules.'$src_lang < $destination_raw'/'$src_lang'_dev.lower' > $valid_en
echo '  '$tgt_lang'...'
subword-nmt apply-bpe -c $destination_raw'/bpe.rules.'$tgt_lang < $destination_raw'/'$tgt_lang'_train.lower' > $train_ru
subword-nmt apply-bpe -c $destination_raw'/bpe.rules.'$tgt_lang < $destination_raw'/'$tgt_lang'_test.lower' > $test_ru
subword-nmt apply-bpe -c $destination_raw'/bpe.rules.'$tgt_lang < $destination_raw'/'$tgt_lang'_dev.lower' > $valid_ru


# create dictionary
echo "creating dictionary..."
cp $train_en $destination_dictionary'/train.'$src_lang
cp $train_ru $destination_dictionary'/train.'$tgt_lang

cp $test_en $destination_dictionary'/test.'$src_lang
cp $test_ru $destination_dictionary'/test.'$tgt_lang

cp $valid_en $destination_dictionary'/valid.'$src_lang
cp $valid_ru $destination_dictionary'/valid.'$tgt_lang

fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
        --trainpref $destination_dictionary'/train' --testpref $destination_dictionary'/test' \
        --validpref $destination_dictionary'/valid' --thresholdtgt 0 --thresholdsrc 0 --workers 20 \
        --destdir $destination_dictionary'/data-dict'  --seed $random_seed

# split train file to smaller files
echo "Spliting ..."
split -l $number_of_line $train_en $destination_split'/'$src_lang'_train.lower.' 
split -l $number_of_line $train_ru $destination_split'/'$tgt_lang'_train.lower.' 

list_split_en=()
list_split_ru=()
for entry in "$destination_split"/*
do
    # echo $entry
    if [[ $entry == *"${src_lang}_train.lower"* ]] ; then
        list_split_en+=($entry) 
    elif [[ $entry == *"${tgt_lang}_train.lower"* ]] ; then
        list_split_ru+=($entry) 
    fi
    
done
# echo "${list_split_en[@]}"


echo "distributing files ..."
count=0
for entry in "${list_split_en[@]}"; do
    # echo "$count"
    # _count=$(printf %03d $count)
    printf -v _count "%02d" $count
    # echo $_count
    name_entry="$(basename $entry)"

    en_file=$destination_split"/"$name_entry
    # echo "----en_file: $en_file"
    # echo "----name_entry: $name_entry"
    # echo "----ru_train: $ru_train"
    ru_file=$destination_split"/"${name_entry/"$src_lang"_train/"$tgt_lang"_train}
    # echo "----ru file: $ru_file"
    mkdir $destination_data_raw"/"$prefix$_count

    cp $en_file $destination_data_raw"/"$prefix$_count'/train.'$src_lang
    echo "\t \t ==== cp $ru_file $destination_data_raw"/"$prefix$_count'/train.'$tgt_lang"
    cp $ru_file $destination_data_raw"/"$prefix$_count'/train.'$tgt_lang
    
    cp $test_en $destination_data_raw"/"$prefix$_count'/test.'$src_lang
    cp $test_ru $destination_data_raw"/"$prefix$_count'/test.'$tgt_lang

    cp $valid_en $destination_data_raw"/"$prefix$_count'/valid.'$src_lang
    cp $valid_ru $destination_data_raw"/"$prefix$_count'/valid.'$tgt_lang
    ((++count))
    fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
        --trainpref $destination_data_raw'/'$prefix$_count'/train' --testpref $destination_data_raw'/'$prefix$_count'/test' \
        --validpref $destination_data_raw'/'$prefix$_count'/valid' --thresholdtgt 0 --thresholdsrc 0 --workers 20 \
        --srcdict $destination_dictionary'/data-dict/dict.'$src_lang'.txt' \
        --tgtdict $destination_dictionary'/data-dict/dict.'$tgt_lang'.txt' \
        --destdir $destination'/data-bin/'$prefix$_count   --seed $random_seed
done
# for dir in "${RESULT[@]}"; do
echo "Done!!!"
exit



echo "fairseq processing..."
fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
    --trainpref $destination_raw'/train' --testpref $destination_raw'/test' \
    --validpref $destination_raw'/valid' --thresholdtgt 0 --thresholdsrc 0 --workers 20 \
    --destdir $destination'/data-bin/'$prefix  --seed  $random_seed
echo "remove some temp files"
# rm -rf $destination_raw'/'$en_train'.lower'
# rm -rf $destination_raw'/'$en_test'.lower'
# rm -rf $destination_raw'/'$en_dev'.lower'
# rm -rf $destination_raw'/'$ru_train'.lower'
# rm -rf $destination_raw'/'$ru_test'.lower'
# rm -rf $destination_raw'/'$ru_dev'.lower'

echo "done!!!"