# directory='/home/cl/huyhien-v/Workspace/MT/data/voita_19/context_agnostic'
directory='/home/cl/huyhien-v/Workspace/MT/data/voita_19/context_aware'
destination='/home/cl/huyhien-v/Workspace/MT/experiments/baseline1_5_last_sent_joined_dict'
# destination='/home/cl/huyhien-v/Workspace/MT/experiments/baseline1_5_full_paragraph'
# destination='/home/cl/huyhien-v/Workspace/MT/experiments/baseline6'
destination_raw=$destination'/raw'
# APPLY_BPE='/home/cl/huyhien-v/Workspace/MT/Voita_19/good-translation-wrong-in-context/lib/tools/apply_bpe.py'
src_lang='en'
tgt_lang='ru'
output_src_tgt=$destination'/output_'$src_lang'-'$tgt_lang
mode=$1 # 'last'

# prefix='baseline1_5m'
prefix='baseline1_5_last_sent_joined_dict'
# prefix='baseline1_5_full_paragraph'
# prefix='baseline6'
# BPE_FOR_VALID_AND_TEST=$1

if [ -d "$destination" ];
then
    echo "$destination exists"
else
    mkdir $destination
fi

if [ -d "$destination_raw" ];
then
    echo "$destination_raw exists"
else
    mkdir $destination_raw
fi

if [ -d "$output_src_tgt" ];
then
    echo "$output_src_tgt exists"
else
    mkdir $output_src_tgt
fi


en_train=$directory'/en_train'
en_test=$directory'/en_test'
en_dev=$directory'/en_dev'
ru_train=$directory'/ru_train'
ru_test=$directory'/ru_test'
ru_dev=$directory'/ru_dev'

# rename and lowercase
train_en=$destination_raw'/train.en'
test_en=$destination_raw'/test.en'
valid_en=$destination_raw'/valid.en'
train_ru=$destination_raw'/train.ru'
test_ru=$destination_raw'/test.ru'
valid_ru=$destination_raw'/valid.ru'

# awk '{print tolower($0)}' < $1 | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $1'.sample'
if [ "$1" = "last" ];
then 
    echo "lowercase and select last sentence... "
    awk '{print tolower($0)}' < $en_train | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $destination_raw'/en_train.lower'
    awk '{print tolower($0)}' < $en_test | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $destination_raw'/en_test.lower'
    awk '{print tolower($0)}' < $en_dev | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $destination_raw'/en_dev.lower'

    awk '{print tolower($0)}' < $ru_train | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $destination_raw'/ru_train.lower'
    awk '{print tolower($0)}' < $ru_test | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $destination_raw'/ru_test.lower'
    awk '{print tolower($0)}' < $ru_dev | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $destination_raw'/ru_dev.lower'
else
    echo "lowercase..."
    awk '{print tolower($0)}' < $en_train > $destination_raw'/en_train.lower'
    awk '{print tolower($0)}' < $en_test > $destination_raw'/en_test.lower'
    awk '{print tolower($0)}' < $en_dev > $destination_raw'/en_dev.lower'

    awk '{print tolower($0)}' < $ru_train > $destination_raw'/ru_train.lower'
    awk '{print tolower($0)}' < $ru_test > $destination_raw'/ru_test.lower'
    awk '{print tolower($0)}' < $ru_dev > $destination_raw'/ru_dev.lower'
fi

echo "subword..."
subword-nmt learn-bpe -s 32000 < $destination_raw'/en_train.lower' > $destination_raw'/bpe.rules.en'
subword-nmt learn-bpe -s 32000 < $destination_raw'/ru_train.lower' > $destination_raw'/bpe.rules.ru'

echo "applying bpe..."
echo "  en..."
# python $APPLY_BPE --bpe_rules $destination_raw'/bpe.rules.en' < $destination_raw'/en_train.lower' > $train_en
# python $APPLY_BPE --bpe_rules $destination_raw'/bpe.rules.en' < $destination_raw'/en_test.lower' > $test_en
# python $APPLY_BPE --bpe_rules $destination_raw'/bpe.rules.en' < $destination_raw'/en_dev.lower' > $valid_en
subword-nmt apply-bpe -c $destination_raw'/bpe.rules.en' < $destination_raw'/en_train.lower' > $train_en
subword-nmt apply-bpe -c $destination_raw'/bpe.rules.en' < $destination_raw'/en_test.lower' > $test_en
subword-nmt apply-bpe -c $destination_raw'/bpe.rules.en' < $destination_raw'/en_dev.lower' > $valid_en
echo "  ru..."
# python $APPLY_BPE --bpe_rules $destination_raw'/bpe.rules.ru' < $destination_raw'/ru_train.lower' > $train_ru
# python $APPLY_BPE --bpe_rules $destination_raw'/bpe.rules.ru' < $destination_raw'/ru_test.lower' > $test_ru
# python $APPLY_BPE --bpe_rules $destination_raw'/bpe.rules.ru' < $destination_raw'/ru_dev.lower' > $valid_ru

subword-nmt apply-bpe -c $destination_raw'/bpe.rules.ru' < $destination_raw'/ru_train.lower' > $train_ru
subword-nmt apply-bpe -c $destination_raw'/bpe.rules.ru' < $destination_raw'/ru_test.lower' > $test_ru
subword-nmt apply-bpe -c $destination_raw'/bpe.rules.ru' < $destination_raw'/ru_dev.lower' > $valid_ru
# if [ "$1" = "1" ]; #Check whether or not we ignore bpe on RU size 
# then
#     cp $destination_raw'/ru_test.lower' $test_ru
#     cp $destination_raw'/ru_dev.lower' $valid_ru
# else
#     python $APPLY_BPE --bpe_rules $destination_raw'/bpe.rules.ru' < $destination_raw'/ru_test.lower' > $test_ru
#     python $APPLY_BPE --bpe_rules $destination_raw'/bpe.rules.ru' < $destination_raw'/ru_dev.lower' > $valid_ru
# fi
echo "fairseq processing..."
fairseq-preprocess --source-lang $src_lang --target-lang $tgt_lang \
    --trainpref $destination_raw'/train' --testpref $destination_raw'/test' \
    --validpref $destination_raw'/valid' --thresholdtgt 0 --thresholdsrc 0 --workers 20 \
    --destdir $destination'/data-bin/'$prefix --joined-dictionary
echo "remove some temp files"
# rm -rf $destination_raw'/'$en_train'.lower'
# rm -rf $destination_raw'/'$en_test'.lower'
# rm -rf $destination_raw'/'$en_dev'.lower'
# rm -rf $destination_raw'/'$ru_train'.lower'
# rm -rf $destination_raw'/'$ru_test'.lower'
# rm -rf $destination_raw'/'$ru_dev'.lower'

echo "done!!!"