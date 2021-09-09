#!/bin/bash
# require the experiment folder:
# - Contain: checkpoints; data-bin; output_en-ru
# - 
# Usage: ./calculate_bleu.sh diretory_to_the_experiment  prefix
#  exp: ./supporting_scripts/calculate_bleu.sh /home/cl/huyhien-v/Workspace/MT/experiments/baseline6 baseline6
directory=$1
prefix=$2
splits=''
if [ "$3" = "split" ];
then
    splits='00'
fi

checkpoints_path=$1'/checkpoints/'$prefix

temp_directory=$1'/output_en-ru/temp' #need to change it for new pair of languages
written_file=$temp_directory'/used_checkpoints.txt'
results_file=$temp_directory'/results.txt'
generate_file=$temp_directory'/generate-test.txt'
used_checkpoints=()
list_check_points=()

#/home/cl/huyhien-v/Workspace/MT/experiments/baseline6/checkpoints/baseline6

# mapfile -t myArray < file.txt

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

condition=1
# while [ $condition -gt 0 ]; do
#     sleep 2m
#     echo "done sleep"
#     # exit
# done
# if (( $(echo "$condition == 1" |bc -l) )); then
#     echo "equal"
# fi
# echo "done"
# exit

while [ $condition -gt 0 ]; do
    
    bash /home/cl/huyhien-v/Workspace/MT/my_fairseq/fairseq/supporting_scripts/delete_checkpoint.sh /home/cl/huyhien-v/Workspace/MT/experiments/baseline6split baseline6split
    for entry in "$checkpoints_path"/*;
    do
        if [[ $entry == *"checkpoint_last"* ]] || [[ $entry == *"checkpoint_best"* ]] ; then
            : 
        else
            list_check_points+=($entry)
        fi
        
    done
    # echo "check0"
    array_diff RESULT list_check_points used_checkpoints
    # echo "===="
    # echo "${used_checkpoints[@]}"
    # echo "===="
    # echo "${list_check_points[@]}"
    # echo "${RESULT[@]}"
    # # fairseq-generate   --batch-size 128 --beam 4 --remove-bpe=subword_nmt --lenpen 0.6 --results-path /home/cl/huyhien-v/Workspace/MT/experiments/baseline6_unbpe_target/output_en-ru

    total_files=${#RESULT[@]}
    echo $total_files
    # echo "check1"
    if (( $(echo "$total_files == 0" |bc -l) )); then
        sleep 5m
        ((++condition))
    fi
    # echo "check2"
    if (( $(echo "$condition == 5" |bc -l) )); then
        echo "Done, exit the script"
        exit
    fi

    count=1
    for dir in "${RESULT[@]}"; do
        
        prefix_arr=(${dir//\// })
        ## echo ${arrIN[1]} 
        ## echo "${prefix_arr[@]}"
        sub_prefix=${prefix_arr[-1]}
        echo "Processing $sub_prefix ... ( $count/$total_files )"
        fairseq-generate $directory/data-bin/$prefix$splits --path $dir --batch-size 128 --beam 4 --remove-bpe=subword_nmt --lenpen 0.6 --results-path $temp_directory
        grep ^H $generate_file | cut -f3- > "$generate_file"."$sub_prefix".sys
        grep ^T $generate_file | cut -f2- > "$generate_file"."$sub_prefix".ref
        output=$(fairseq-score --sys $generate_file"."$sub_prefix".sys" --ref $generate_file"."$sub_prefix".ref")
        # echo "$output"
        final_result="${output##*$'\n'}"
        echo "$sub_prefix -- $final_result" >> $results_file

        echo $dir >> $written_file
        ((++count))
        # break
    done
done
echo " Done!!!"
# ./supporting_scripts/calculate_bleu.sh /home/cl/huyhien-v/Workspace/MT/experiments/baseline6split baseline6split split