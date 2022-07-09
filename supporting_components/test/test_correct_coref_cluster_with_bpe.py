import sys, os

from src.correct_coref_cluster_with_bpe import correct_cluster_with_bpe


input_sentence = "It says he was under your care . Who is this ? How is he ? He 's not there ." 
input_cluster = "[[[2, 2], [14, 14], [16, 16]]]"

input_sent_eos = "It says he was under your care . _eos Who is this ? _eos How is he ? _eos He 's not there ." 
expected_clusters_eos = "[[[2, 2], [16, 16], [19, 19]]]"

input_sent_bpe = "It says he was un@@ der your care . _eos W@@ ho is this ? _eos How is he ? _eos He 's not th@@ ere ." 
expected_bpe_clusters = "[[[2, 2], [18, 18], [21, 21]]]"

# from pudb import set_trace; set_trace()
assert expected_bpe_clusters == correct_cluster_with_bpe(input_sent_eos, input_sent_bpe, expected_clusters_eos)
print ('pass')

input_sentence = "So ? _eos I need insurance . _eos You have insurance . _eos So I thought maybe you could marry me ." 
input_cluster = "[[[3, 3], [14, 14], [20, 20]], [[8, 8], [17, 17]]]"
input_sent_eos = "So ? _eos I need insurance . _eos You have insurance . _eos So I thought maybe you could marry me ." 
expected_clusters = "[[[3, 3], [14, 14], [20, 20]], [[8, 8], [17, 17]]]"
assert expected_clusters == correct_cluster_with_bpe(input_sentence, input_sent_eos, input_cluster)

input_sentence = "Is she in danger ? _eos No . _eos Things are different there . _eos Different ?" 
input_cluster = "[]"

input_sent_eos = "Is she in danger ? _eos No . _eos Things are different there . _eos Different ?" 
expected_clusters = "[]"
assert expected_clusters == correct_cluster_with_bpe(input_sentence, input_sent_eos, input_cluster)
print('pass')