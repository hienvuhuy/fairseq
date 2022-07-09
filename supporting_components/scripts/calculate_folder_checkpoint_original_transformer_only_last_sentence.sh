#!/bin/bash

# Usage
#   bash calculate_folder_checkpoint.sh     checkpoint_folder_path     data_path    output_path    outputname
#   outputname: to distinguish different experiments
#   Ex: bash calculate_folder_checkpoint.sh /home/cl/huyhien-v/Workspace/MT/experiments/baseline6mdepcor /home/is/huyhien-v/Data/Bin/MT/en-ru/voita-19/baseline6m_bpe6m_split_400k/data-bin/part00 /var/autofs/cl/work/huyhien-v/Workspace/MT/experiments  baseline6m_origin_depcor_wo_matrix
#   Ex: bash calculate_folder_checkpoint.sh /home/cl/huyhien-v/Workspace/MT/experiments/baseline6mdepcor /home/is/huyhien-v/Data/Bin/MT/en-ru/voita-19/baseline6m_bpe6m_split_400k/data-bin/part00 /var/autofs/cl/work/huyhien-v/Workspace/MT/experiments/tmp baseline6mdepcor



# June 14:
#   Run: bash /home/cl/huyhien-v/Workspace/MT/my_fairseq/probing/fairseq-probing/supporting_components/scripts/calculate_folder_checkpoint.sh /home/cl/huyhien-v/Workspace/MT/experiments/translation_coref_probing/baseline_probing /home/is/huyhien-v/Data/Bin/MT/en-ru/coref-mt-probing-bos/bin  /home/cl/huyhien-v/Workspace/MT/experiments/RESULTs/probings  basline_probing

#   Test: fairseq-generate /home/is/huyhien-v/Data/Bin/MT/en-ru/coref-mt-probing-bos/bin --path /home/cl/huyhien-v/Workspace/MT/experiments/translation_coref_probing/baseline_probing/checkpoint_best.pt --beam 4 --batch-size 128 --task translation_coref_probing --load-source-coref true --coref-cluster-path /home/is/huyhien-v/Data/Bin/MT/en-ru/coref-mt-probing-bos/cluster_with_bpe/ --results-path /home/cl/huyhien-v/Workspace/MT/experiments/RESULTs/probings/baseline_probing


## For original:

task='translation'#dep
# task='translation_coref_probing'

checkpoints_path=$1
data_path=$2

prefix=$4

if [ -z "$4" ]; then
    prefix='output'
fi

echo "Checkpoint path $checkpoints_path"

# temp_directory=$1'/output_en-ru/temp' #need to change it for new pair of languages
temp_directory=$3'/'$prefix
written_file=$temp_directory'/used_checkpoints.txt'
results_file=$temp_directory'/results.txt'
results_lastsent_file=$temp_directory'/results.lastsent.txt'
generate_file=$temp_directory'/generate-test.txt'

used_checkpoints=()
list_check_points=()

echo "$temp_directory"

if [ -d "$temp_directory" ];
then
    :
else
    mkdir $temp_directory
fi

if [ -f "$written_file" ];
then
    mapfile -t used_checkpoints < $written_file
fi


function array_diff {
    eval local ARR1=\(\"\${$2[@]}\"\)
    eval local ARR2=\(\"\${$3[@]}\"\)
    local IFS=$'\n'
    mapfile -t $1 < <(comm -23 <(echo "${ARR1[*]}" | sort) <(echo "${ARR2[*]}" | sort))
}


for entry in "$checkpoints_path"/*
do
    echo "$entry"
    if [[ $entry == *"checkpoint_last"* ]] || [[ $entry == *"checkpoint_best"* ]] ; then
        : 
    else
        list_check_points+=($entry)
    fi
    
done

array_diff RESULT list_check_points used_checkpoints
# echo "===="
# echo "${used_checkpoints[@]}"
# echo "===="
# echo "${list_check_points[@]}"
# echo "${RESULT[@]}"
# # fairseq-generate   --batch-size 128 --beam 4 --remove-bpe=subword_nmt --lenpen 0.6 --results-path /home/cl/huyhien-v/Workspace/MT/experiments/baseline6_unbpe_target/output_en-ru

total_files=${#RESULT[@]}
echo "Total checkpoints: $total_files."


count=1
for dir in "${RESULT[@]}"; do


    prefix_arr=(${dir//\// })
    sub_prefix=${prefix_arr[-1]}
    
    echo "Processing $sub_prefix ... ( $count/$total_files )"
    if [[ ! -f $dir ]]
    then
        echo "      It is folder, ignore!!"
        continue
    fi

    # echo "fairseq-generate $data_path --path $dir --batch-size 128 --beam 4 --remove-bpe=subword_nmt --lenpen 0.6 --results-path $temp_directory"
    fairseq-generate $data_path --path $dir --batch-size 128 --beam 4 --remove-bpe=subword_nmt --lenpen 0.6 --results-path $temp_directory \
    # --task translation_coref_probing --load-source-coref --coref-cluster-path "/home/is/huyhien-v/Data/Bin/MT/en-ru/coref-mt-probing-bos/cluster_with_bpe/"
    
    # echo 
    # exit
    # grep ^H $generate_file | cut -f3- > "$generate_file"."$sub_prefix".sys
    grep ^H $generate_file | cut -f3- | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $generate_file.$sub_prefix.sys
    # grep ^T $generate_file | cut -f2- > "$generate_file"."$sub_prefix".ref
    grep ^T $generate_file | cut -f2- | awk '{split($0, a, "_eos"); print a[4]}' | awk '{$1=$1};1' > $generate_file.$sub_prefix.ref
    
    output=$(fairseq-score --sys $generate_file"."$sub_prefix".sys" --ref $generate_file"."$sub_prefix".ref")
    
    # echo "$output"
    final_result="${output##*$'\n'}"
    echo "$sub_prefix -- $final_result" >> $results_file

    echo $dir >> $written_file
    ((++count))
done
echo "Done!!!"