echo "Export translated sentences and reference sentences"
echo "Input: translation file of fairseq"
# S: source
# T: target
# H: Hypothesis (system)
# D: Hypothesis with replace BPE, tokenizer
# check link: https://github.com/pytorch/fairseq/issues/3000
num=$2
grep ^H $1 | cut -f3- > "$1"."$2".sys
grep ^T $1 | cut -f2- > "$1"."$2".ref

# grep ^H $1 | awk -F'_eos' {'print$4'} | awk '{$1=$1}1' > "$1"."$2".last.sys
# grep ^T $1 | awk -F'_eos' {'print$4'} | awk '{$1=$1}1' > "$1"."$2".last.ref

echo "Output of full paragraph"
# fairseq-score --sys "$1".sys --ref "$1".ref --sacrebleu
# fairseq-score --sys "$1".last.sys --ref "$1".last.ref --sacrebleu
echo "  command: fairseq-score --sys $1.$2.sys --ref $1.$2.ref"
fairseq-score --sys "$1"."$2".sys --ref "$1"."$2".ref 

# echo "Output of last sentence"
# echo "  command: fairseq-score --sys $1.$2.last.sys --ref $1.$2.last.ref"
# fairseq-score --sys "$1"."$2".last.sys --ref "$1"."$2".last.ref 

echo "DONE!!"