import sys, ast


# input_sentence = "It says he was under your care . Who is this ? How is he ? He 's not there ." 
# input_cluster = "[[[2, 2], [14, 14], [16, 16]]]"

# input_sent_eos = "It says he was under your care . _eos Who is this ? _eos How is he ? _eos He 's not there ." 
# expected_clusters_eos = "[[[2, 2], [16, 16], [19, 19]]]"

# input_sent_bpe = "It says he was un@@ der your care . _eos W@@ ho is this ? _eos How is he ? _eos He 's not th@@ ere ." 
# expected_bpe_clusters = "[[[2, 2], [18, 18], [21, 21]]]"

def bpe_solve(original, bpestr, separator='@@'):
    list_original = original.strip().split()
    list_bpestr = bpestr.strip().split()
    dict_map_word_2_bpe = {}
    dict_map_bpe_2_word = {}
    # list_bpe = [remove_bpe(i) for i in bpestr.strip().split()]
    index_original = 0
    id_word, word, words = [], [], []
    for idx, tok in enumerate(list_bpestr):
        if tok.endswith(separator):
            tok = tok.strip(separator)
            word.append(tok)
            id_word.append(idx)
            dict_map_bpe_2_word[idx] = index_original
        else:
            word.append(tok)
            tmp = ''.join(word)
            # from pudb import set_trace; set_trace()
            # assert tmp == list_original[index_original]
            assert tmp.lower() == list_original[index_original].lower()
            id_word.append(idx)
            dict_map_word_2_bpe[index_original] = id_word
            dict_map_bpe_2_word[idx] = index_original
            index_original+=1
            word = []
            id_word = []
    return dict_map_word_2_bpe, dict_map_bpe_2_word

def correct_cluster_with_bpe(input_sent_wo_bpe, input_sent_w_bpe, original_cluster_str):
    """
        Convert coref-clusters of sentences without _eos to coref-cluster of sentences with _eos
        Input:  a sentence without _eos
                a cluster of sentence without _eos
                a sentence with _eos
        Output: a cluster of sentence with _eos

    """
    coref_cluster = ast.literal_eval(original_cluster_str.strip())
    
    dict_word_2_bpe, dict_bpe_2_word = bpe_solve(input_sent_wo_bpe, input_sent_w_bpe)

    bpe_coref_cluster = []
    for cluster in coref_cluster:
        _tp_cluster = []
        for member in cluster:
            _new_member = [min(dict_word_2_bpe.get(member[0])), max(dict_word_2_bpe.get(member[1]))]
            _tp_cluster.append(_new_member)
        bpe_coref_cluster.append(_tp_cluster)
    
    return str(bpe_coref_cluster)

# assert expected_bpe_clusters == correct_cluster_with_bpe(input_sent_eos, input_sent_bpe, expected_clusters_eos)
# print ('pass')
# sys.exit()
# input_sentence = "So ? I need insurance . You have insurance . So I thought maybe you could marry me ." 
# input_cluster = "[[[2, 2], [11, 11], [17, 17]], [[6, 6], [14, 14]]]"

# input_sent_eos = "So ? _eos I need insurance . _eos You have insurance . _eos So I thought maybe you could marry me ." 
# expected_clusters = "[[[3, 3], [14, 14], [20, 20]], [[8, 8], [17, 17]]]"
# assert expected_clusters == correct_cluster_with_bpe(input_sentence, input_cluster, input_sent_eos)



# input_sentence = "Is she in danger ? No . Things are different there . Different ?" 
# input_cluster = "[]"

# input_sent_eos = "Is she in danger ? _eos No . _eos Things are different there . _eos Different ?" 
# expected_clusters = "[]"
# assert expected_clusters == correct_cluster_with_bpe(input_sentence, input_cluster, input_sent_eos)
# print('pass')