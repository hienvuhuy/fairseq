
## Command:
### Notes:
* Go to experiments directory and call exact commands for an exact experiment
### En-Ru
* fairseq-preprocess --source-lang en --target-lang ru --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/voita.en-ru --thresholdtgt 0 --thresholdsrc 0 --workers 20
* CUDA_VISIBLE_DEVICES=0,1 fairseq-train data-bin/voita.en-ru --arch transformer_en_ru --dropout 0.2 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer nag --clip-norm 0.1 --lr 0.5 --lr-scheduler fixed --force-anneal 50 --max-tokens 4000 --save-dir checkpoints/voita.en-ru --keep-last-epochs 10 --batch-size 10

* CUDA_VISIBLE_DEVICES=0,1 fairseq-train data-bin/voita.en-ru --arch transformer_wmt_en_de --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.1 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4000 --keep-last-epochs 10 --eval-bleu --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --save-dir checkpoints/voita.en-ru.new --num-workers 4 --model-parallel-size 2

### Normal MT
<!-- * TEXT=examples/translation/wmt17_en_de
* fairseq-preprocess --source-lang en --target-lang de --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0 --workers 20
* CUDA_VISIBLE_DEVICES=0,1 fairseq-train data-bin/wmt17_en_de --arch fconv_wmt_en_de --dropout 0.2 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer nag --clip-norm 0.1 --lr 0.5 --lr-scheduler fixed --force-anneal 50 --max-tokens 4000 --save-dir checkpoints/fconv_wmt_en_de --keep-last-epochs 10 --scoring bleu -->



#### Download
* cd examples/translation/
* bash prepare-iwslt14.sh
* cd ../..

#### Preprocess/binarize the data
* TEXT=examples/translation/iwslt14.tokenized.de-en
* fairseq-preprocess --source-lang de --target-lang en --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/iwslt14.tokenized.de-en --workers 20

* CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --keep-last-epochs 10 --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric 

CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en --arch transformer_wmt_en_de --share-decoder-input-output-embed --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --keep-last-epochs 10 --eval-bleu --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --save-dir checkpoints/transformer-en-de
themes/pygmalion.zsh-theme