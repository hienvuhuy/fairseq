## Debug on faiseq
* Use with Cpu only
* Command: 
  * fairseq-preprocess --source-lang en --target-lang vi --trainpref /home/cl/huyhien-v/Workspace/MT/data/debug/train --validpref /home/cl/huyhien-v/Workspace/MT/data/debug/valid --destdir data-bin/debug --thresholdtgt 0 --thresholdsrc 0 --cpu
  * fairseq-train --cpu data-bin/debug --arch transformer_wmt_en_de --optimizer adam --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.1 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 500 --keep-last-epochs 10 --eval-bleu --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --save-dir checkpoints --num-workers 1 

## Note

* Dictionary of fairseq.data.dictionary.Dictionary is a dict to store mapping word to id
* Check this one for some example:
  * https://www.jianshu.com/p/3a106a9fb44b
  * https://yinghaowang.xyz/technology/2020-03-14-FairseqTransformer.html

* For printing validation results while training:
  * use option:
    * --eval-bleu-print-samples
    * Code in : task/translation.py
        * if self.cfg.eval_bleu_print_samples:
            from pudb import set_trace; set_trace()