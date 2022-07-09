"""
    Filter bilingual corpus with coref information (text without _eos)
    Return:
        - bilingual text with _eos and contains correct linked clusters
        - coref clusters with correcting info (since we add _eos)

"""

# filter sentences which possess connection to the context 

import sys
import os
import ast

def correct_cluster_with_eos(input_sent_wo_eos, original_cluster_str, input_sent_w_eos):
    """
        Convert coref-clusters of sentences without _eos to coref-cluster of sentences with _eos
        Input:  a sentence without _eos
                a cluster of sentence without _eos
                a sentence with _eos
        Output: a cluster of sentence with _eos

    """
    list_word_wo_eos = input_sent_wo_eos.strip().split()
    list_word_w_eos = input_sent_w_eos.strip().split()
    
    dict_word_wo_eos_to_idx = {}

    dict_map = {}
    dict_idx = {}
    w_eos_idx = 0
    break_check = 0
    for idx, word in enumerate(list_word_wo_eos):
        while word != list_word_w_eos[w_eos_idx]:
            w_eos_idx+=1
            if break_check==10:
                print('something\'s wrong')
                sys.exit()

        
        dict_map[word+'-|-'+str(idx)] = word+'-|-'+str(w_eos_idx)
        dict_idx[idx] = w_eos_idx

    dict_cluster_eos = {}
    cluster_wo_eos = ast.literal_eval(original_cluster_str.strip())
    expected_clusters = []
    for cluster in cluster_wo_eos:
        temp_cluster = []
        for item in cluster:
            _temp_content = []
            for _content in item:
                _temp_content.append(dict_idx.get(_content))
            temp_cluster.append(_temp_content)
        expected_clusters.append(temp_cluster)
    
    return str(expected_clusters)


def is_relevant_to_last_sentence(sents, sents_with_eos, coref_cluster):
    status = True
    list_sents = sents_with_eos.strip().split('_eos')
    list_words = sents.strip().split()
    # list_sents = sents.strip().split()
    min_idx_to_last_sentence = sum([len(i.strip().split()) for i in list_sents[:-1]])
    for cluster in coref_cluster:
        contain_upper = False
        contain_lower = False
        for item in cluster:
            for _idx in range(item[0], item[-1]+1):
                if _idx > min_idx_to_last_sentence:
                    contain_upper = True
                if _idx < min_idx_to_last_sentence:
                    contain_lower = True
            if contain_upper == True and contain_lower == True:
                return True
    return False

out_directory = '/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware_with_target/sample_check/'
relevant_directory = out_directory + 'relevant/'
irrelevant_directory = out_directory + 'irrelevant/'



# ########### test ########
# # coref_part1 = '/home/cl/huyhien-v/Workspace/MT/my_fairseq/recheck/fairseq-feat/manually-checking/allennlp/retag/en_test_voita.txt'
# coref_part1 = '/home/cl/huyhien-v/Workspace/MT/my_fairseq/probing/fairseq-probing/supporting_components/output/coref/en_test_voita.txt'

# list_coref_sents = [i.strip() for i in open(coref_part1, 'r').readlines()]
# original_data_en = '/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware/en_test'
# original_data_ru = '/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware/ru_test'

# ## output for train
# out_relevant_en = open(relevant_directory+'en_test', 'w')
# out_relevent_en_cluster = open(relevant_directory+'en_test_coref', 'w')
# out_relevant_ru = open(relevant_directory+'ru_test', 'w')
# out_irrelevant_en = open(irrelevant_directory+'en_test', 'w')
# out_irrelevant_ru = open(irrelevant_directory+'ru_test', 'w')
# out_related_list = open(relevant_directory+'/test_list_idx.txt', 'w')


# ########### train ########
# # coref_part1 = '/home/cl/huyhien-v/Workspace/MT/my_fairseq/recheck/fairseq-feat/manually-checking/allennlp/retag/en_train_voita.txt'
# coref_part1 = '/home/cl/huyhien-v/Workspace/MT/my_fairseq/probing/fairseq-probing/supporting_components/output/coref/en_train_voita.txt'
# list_coref_sents = [i.strip() for i in open(coref_part1, 'r').readlines()]
# original_data_en = '/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware/en_train'
# original_data_ru = '/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware/ru_train'

# ## output for train
# out_relevant_en = open(relevant_directory+'en_train', 'w')
# out_relevent_en_cluster = open(relevant_directory+'en_train_coref', 'w')
# out_relevant_ru = open(relevant_directory+'ru_train', 'w')
# out_irrelevant_en = open(irrelevant_directory+'en_train', 'w')
# out_irrelevant_ru = open(irrelevant_directory+'ru_train', 'w')
# out_related_list = open(relevant_directory+'/train_list_idx.txt', 'w')


########### dev ########
# coref_part1 = '/home/cl/huyhien-v/Workspace/MT/my_fairseq/recheck/fairseq-feat/manually-checking/allennlp/retag/en_dev_voita.txt'
coref_part1 = '/home/cl/huyhien-v/Workspace/MT/my_fairseq/probing/fairseq-probing/supporting_components/output/coref/en_dev_voita.txt'
list_coref_sents = [i.strip() for i in open(coref_part1, 'r').readlines()]
original_data_en = '/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware/en_dev'
original_data_ru = '/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware/ru_dev'

## output for dev
out_relevant_en = open(relevant_directory+'en_dev', 'w')
out_relevent_en_cluster = open(relevant_directory+'en_dev_coref', 'w')
out_relevant_ru = open(relevant_directory+'ru_dev', 'w')
out_irrelevant_en = open(irrelevant_directory+'en_dev', 'w')
out_irrelevant_ru = open(irrelevant_directory+'ru_dev', 'w')
out_related_list = open(relevant_directory+'/dev_list_idx.txt', 'w')



##### process data

list_original_en = [i.strip() for i in open(original_data_en).readlines()]
list_original_ru = [i.strip() for i in open(original_data_ru).readlines()]


print(len(list_original_en), ' vs ', len(list_coref_sents))
assert len(list_original_en) == len(list_coref_sents)


# related_list: list of indices of senteces in original dataset.
# for example: if sentence 4 (the first sentence) in original dataset contains linked coref clusters. its index in original is 4; its news index in results is 0
related_list = []
for idx, item in enumerate(list_original_en):
    # from pudb import set_trace; set_trace()
    # print(idx)
    [sents, coref] = [i.strip() for i in list_coref_sents[idx].strip().split('|||')]
    list_sents = [i.strip() for i in sents.strip().split()]
    list_coref = ast.literal_eval(coref.strip())
    # from pudb import set_trace; set_trace()
    if is_relevant_to_last_sentence(sents, item, list_coref):
        out_relevant_en.write(list_original_en[idx].strip()+'\n')
        out_relevant_ru.write(list_original_ru[idx].strip()+'\n')
        # out_relevent_en_cluster.write(coref.strip()+'\n')
        out_relevent_en_cluster.write(correct_cluster_with_eos(sents, coref, list_original_en[idx])+'\n')
        related_list.append(idx)
    else:
        out_irrelevant_en.write(list_original_en[idx].strip()+'\n')
        out_irrelevant_ru.write(list_original_ru[idx].strip()+'\n')

out_relevant_ru.close()
out_relevant_en.close()

print(len(related_list))

for i in related_list:
    out_related_list.write(str(i).strip()+'\n')
out_related_list.close()
