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

## Distributed training
* Check SLURM
* https://fairseq.readthedocs.io/en/latest/getting_started.html#distributed-training



* Sample function from Zhang et al. 2020
```
def generate_masking(inputs, sentence_sep_id):
	"""GENERATE LONG SHORT TERM MASKING
		ARGS:
			INPUTS: A DENSE VECTOR  [BATCH, LENGTH] OF
				SOURCE OR TARGET WORD IDS
			SENTENCE_SEP_ID: THE ID OF THE SENTENCE
				SEPARATION TOKEN
	"""
	shape = tf.shape(inputs)
	length = shape[1]
	
	sentence_sep_id_matrix = sentence_sep_id * tf.ones(shape, dtype=inputs.dtype)
	sentence_end = tf.cast(tf.equal(inputs, sentence_sep_id), tf.float32)
	sentence_end_mask = tf.cumsum(sentence_end, axis = -1)
	
	sentence_end_mask_expand_row = tf.expand_dims(sentence_end_mask, -1)
	sentence_end_mask_expand_row = tf.tile(sentence_end_mask_expand_row, [1, 1,length])
	
	sentence_end_mask_expand_column = tf.expand_dims(sentence_end_mask, -2)
	sentence_end_mask_expand_column = tf.tile(sentence_end_mask_expand_column, [1,length, 1])
	
	mask = tf.cast(tf.equal(sentence_end_mask_expand_row,sentence_end_mask_expand_column), tf.float32)
	mask = -1e9 * (1.0 - mask)
	mask = tf.reshape(mask, [-1, 1, length,length])
	
	return mask
```


## Some notes about dataset in fairseq
* In task (my_translation.py): 
  * self.src_dict (fairseq.data.dictionary)
    * method string(Tensor) -> return string of input tensor
    * method index(sym)  -> return index of specify tensor
    * if you want to check what character in the batch:
      * self.src_dict[Tensor with dim torch.Size([]) or torch.Size([1]) ]
  * self.tgt_dict (fairseq.data.dictionary)
    * Similar to self.src_dict

* Criterion:
  * Compute loss