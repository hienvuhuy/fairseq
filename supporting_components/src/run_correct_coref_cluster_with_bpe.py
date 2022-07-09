from correct_coref_cluster_with_bpe import correct_cluster_with_bpe
from tqdm import trange

output_directory = "/home/is/huyhien-v/Data/Bin/MT/en-ru/coref-mt-probing-bos/cluster_with_bpe/"
input_directory = "/home/is/huyhien-v/Data/Bin/MT/en-ru/coref-mt-probing-bos"

bpe_inputs_directory    = input_directory+"/bpe/"
wo_bpe_inputs_directory = input_directory+"/lower/"
coref_wo_bpe_directory  = "/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware_with_target/sample_check/relevant_add_bos/"
# coref_wo_bpe_directory  = "/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware_with_target/sample_check/relevant/"

##### Training
path_to_bpe_input = bpe_inputs_directory+"train.en"
path_to_wo_bpe_input = wo_bpe_inputs_directory+"en_train.lower"
path_to_coref_info_input = coref_wo_bpe_directory+"en_train_coref"
# coref_w_bpe_output = output_directory+"en_train_coref_w_bpe"
coref_w_bpe_output = output_directory+"train.en-ru.en.coref"

# ##### Valid
# path_to_bpe_input = bpe_inputs_directory+"valid.en"
# path_to_wo_bpe_input = wo_bpe_inputs_directory+"en_dev.lower"
# path_to_coref_info_input = coref_wo_bpe_directory+"en_dev_coref"
# #coref_w_bpe_output = output_directory+"en_dev_coref_w_bpe"
# coref_w_bpe_output = output_directory+"valid.en-ru.en.coref"

# ##### Testing
# path_to_bpe_input = bpe_inputs_directory+"test.en"
# path_to_wo_bpe_input = wo_bpe_inputs_directory+"en_test.lower"
# path_to_coref_info_input = coref_wo_bpe_directory+"en_test_coref"
# # coref_w_bpe_output = output_directory+"en_test_coref_w_bpe"
# coref_w_bpe_output = output_directory+"test.en-ru.en.coref"



bpe_inputs = [i.strip() for i in open(path_to_bpe_input, 'r').readlines()]
wo_bpe_inputs = [i.strip() for i in open(path_to_wo_bpe_input, 'r').readlines()]
coref_wo_bpe_inputs = [i.strip() for i in open(path_to_coref_info_input, 'r').readlines()]
outfile_coref = open(coref_w_bpe_output, 'w')

assert len(bpe_inputs) == len(wo_bpe_inputs)
assert len(bpe_inputs) == len(coref_wo_bpe_inputs)


for idx in trange(len(bpe_inputs)):
    # new_coref_info = correct_cluster_with_bpe(input_sent_wo_bpe, input_sent_w_bpe, original_cluster_str)
    new_coref_info = correct_cluster_with_bpe(wo_bpe_inputs[idx], bpe_inputs[idx], coref_wo_bpe_inputs[idx])
    outfile_coref.write(new_coref_info.strip()+'\n')

outfile_coref.close()

print('Done')
