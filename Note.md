## Notes about develope this tool 
* /HOME/ is the location of project fairseq
* [x] Add VERSION.md to specify your fairseq verion.
* [ ] Add your new implementation in /HOME/fairseq/fairseq/data
* [ ] Register your own model in /HOME/fairseq/fairseq/model
* [ ] Do a tutorial 

* Follow [this repo](https://github.com/e-bug/pascal) to know the real project 

## Build a model and modules for new translation experiments
### Build a baseline
* fairseq/models/transformer_original_based_model.py
### Build co-refernce experiments
* Need:
  * fairseq/data/coreference_tag_dataset.py
  * fairseq/models/transformers_coreference.py
  * fairseq/models/fairseq_model.py (edit and register the model)
  * fairseq/modules/multihead_coreference.py
  * task/coreference_translation.py
## Command:
* fairseq-preprocess --source-lang en --target-lang ru --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/voita.en-ru --thresholdtgt 0 --thresholdsrc 0 --workers 20
* CUDA_VISIBLE_DEVICES=0,1 fairseq-train data-bin/voita.en-ru --arch transformer_en_ru --dropout 0.2 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --optimizer nag --clip-norm 0.1 --lr 0.5 --lr-scheduler fixed --force-anneal 50 --max-tokens 4000 --save-dir checkpoints/voita.en-ru --keep-last-epochs 10 --batch-size 10