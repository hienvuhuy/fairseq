# go to the root of the test folder and run
#   python -m test.test_correct_coref_cluster_with_bos

import sys, ast



import sys, os
from src.correct_coref_cluster_with_bos import correct_cluster_with_bos, add_bos_into_cluster

input_sentence = "It says he was under your care . _eos Who is this ? _eos How is he ? _eos He 's not there ." 
input_cluster = "[[[2, 2], [16, 16], [19, 19]]]"

input_sent_bos = "_bos It says he was under your care . _eos Who is this ? _eos How is he ? _eos He 's not there ." 
expected_clusters = "[[[3, 3], [17, 17], [20, 20]]]"

assert expected_clusters == correct_cluster_with_bos(input_sentence, input_sent_bos, input_cluster)
assert expected_clusters == add_bos_into_cluster(input_cluster)


input_sentence = "It says he was under your care . Who is this ? How is he ? He 's not there ." 
input_cluster = "[[[2, 2], [14, 14], [16, 16]]]"

input_sent_bos = "_bos It says he was under your care . Who is this ? How is he ? He 's not there ." 
expected_clusters = "[[[3, 3], [15, 15], [17, 17]]]"

assert expected_clusters == correct_cluster_with_bos(input_sentence, input_sent_bos, input_cluster)
assert expected_clusters == add_bos_into_cluster(input_cluster)

input_sentence = "It says he was under your care . Who is this ? How is he ? He 's not there ." 
input_cluster = "[[[2, 4], [14, 14], [16, 18]]]"

input_sent_bos = "_bos It says he was under your care . Who is this ? How is he ? He 's not there ." 
expected_clusters = "[[[3, 5], [15, 15], [17, 19]]]"

assert expected_clusters == correct_cluster_with_bos(input_sentence, input_sent_bos, input_cluster)
assert expected_clusters == add_bos_into_cluster(input_cluster)
print(input_cluster)
print(add_bos_into_cluster(input_cluster))
print(expected_clusters)
print("pass")