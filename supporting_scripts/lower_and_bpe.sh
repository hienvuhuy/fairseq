# Run: lower_and_bpe.sh input_file
# Output: input_file with bpe, remember that, we delete the original file

input=$1
apply_bpe="/home/cl/huyhien-v/Workspace/MT/my_fairseq/fairseq/supporting_scripts/apply_bpe.py "
#Lowercase
command_lowercase="cat $input | python3 -c 'import sys; print(sys.stdin.read().lower())' > $input.lower"
command_subword_nmt="subword-nmt learn-bpe -s 32000 < $input.lower > $input.bpe_rules"
command_bpe="$apply_bpe --bpe_rules $input.bpe_rules < $input.lower > $input.lower.bpe"
# remove_unnecessary_file="rm -rf $input $input.lower && mv $input.lower.bpe $input"

echo "Running: $command_lowercase .."
$command_lowercase

echo "Running: $command_subword_nmt .."
$command_subword_nmt

echo "Running: $command_bpe .."
$command_bpe