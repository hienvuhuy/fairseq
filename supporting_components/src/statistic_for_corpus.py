import sys, ast

# file_in = '/home/is/huyhien-v/Data/Bin/MT/en-ru/coref-mt-probing/cluster_with_bpe/en_test_coref_w_bpe'
# file_in = '/home/is/huyhien-v/Data/Bin/MT/en-ru/coref-mt-probing/cluster_with_bpe/en_dev_coref_w_bpe'
file_in = '/home/is/huyhien-v/Data/Bin/MT/en-ru/coref-mt-probing/cluster_with_bpe/en_train_coref_w_bpe'

list_idx = [i.strip() for i in open(file_in, 'r').readlines()]
numbers_of_clusters = 0
max_number_of_clusters = 0
max_items_in_cluster = 0
max_number_of_words_in_cluster = 0
for item in list_idx:
    list_clusters = ast.literal_eval(item.strip())
    # from pudb import set_trace; set_trace()
    if len(list_clusters) > max_number_of_clusters:
        max_number_of_clusters = len(list_clusters)
    for cluster in list_clusters:
        _all_words = 0
        for _c in cluster:
            begin, end = _c
            _all_words += (end - begin) +1
        if max_number_of_words_in_cluster < _all_words:
            max_number_of_words_in_cluster = _all_words

        if len(cluster) > max_items_in_cluster:
            max_items_in_cluster = len(cluster)
    numbers_of_clusters+=len(list_clusters)
    

# from pudb import set_trace; set_trace()
average = numbers_of_clusters/len(list_idx)
print(average)
print(max_number_of_clusters)
print(max_items_in_cluster)
print(max_number_of_words_in_cluster)