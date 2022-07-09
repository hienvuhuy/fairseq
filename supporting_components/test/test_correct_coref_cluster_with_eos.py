import sys, os
# from ..src.correct_coref_cluster_with_eos import correct_cluster_with_eos
from src.correct_coref_cluster_with_eos import correct_cluster_with_eos

input_sentence = "It says he was under your care . Who is this ? How is he ? He 's not there ." 
input_cluster = "[[[2, 2], [14, 14], [16, 16]]]"

input_sent_eos = "It says he was under your care . _eos Who is this ? _eos How is he ? _eos He 's not there ." 
expected_clusters = "[[[2, 2], [16, 16], [19, 19]]]"

assert expected_clusters == correct_cluster_with_eos(input_sentence, input_sent_eos, input_cluster)


input_sentence = "So ? I need insurance . You have insurance . So I thought maybe you could marry me ." 
input_cluster = "[[[2, 2], [11, 11], [17, 17]], [[6, 6], [14, 14]]]"

input_sent_eos = "So ? _eos I need insurance . _eos You have insurance . _eos So I thought maybe you could marry me ." 
expected_clusters = "[[[3, 3], [14, 14], [20, 20]], [[8, 8], [17, 17]]]"
assert expected_clusters == correct_cluster_with_eos(input_sentence, input_sent_eos, input_cluster)



input_sentence = "Is she in danger ? No . Things are different there . Different ?" 
input_cluster = "[]"

input_sent_eos = "Is she in danger ? _eos No . _eos Things are different there . _eos Different ?" 
expected_clusters = "[]"
assert expected_clusters == correct_cluster_with_eos(input_sentence, input_sent_eos, input_cluster)
print('pass')