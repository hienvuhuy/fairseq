#!/bin/bash
#usage: ./delete_checkpoint.sh experiment_path prefix

threshold=32.01
directory=$1
prefix=$2
checkpoints=$1"/checkpoints/$prefix"
ketqua=$1"/output_en-ru/temp/results.txt" #need to change it (en-ru) for new pairs of languages
list_used_check_points=()

mapfile -t list_used_check_points < $ketqua
for dir in "${list_used_check_points[@]}"; do
    # echo "$dir"
    if [[ $dir == *"-- BLEU4 ="* ]]; then
        # echo "$dir"
        name=(${dir//--/ })
        name=${name[0]}
        score=(${dir//,/ })
        score=${score[4]}
        score=$(echo "scale=2; $score" | bc)
        # echo $name
        # echo $score
        
        if (( $(echo "$score < $threshold" |bc -l) )); then
            if [ -f "$checkpoints/$name" ];
            then
                echo "remove $checkpoints/$name"
                rm -rf $checkpoints"/"$name
            fi
            # echo "remove $checkpoints/$name"
            # # echo "rm -rf $checkpoints/$name"
            # rm -rf $checkpoints"/"$name
        fi
    fi
# exit
done
echo "Done!!"

# ./supporting_scripts/delete_checkpoint.sh /home/cl/huyhien-v/Workspace/MT/experiments/baseline6split baseline6split