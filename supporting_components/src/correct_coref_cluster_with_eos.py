import sys, ast

def correct_cluster_with_eos(input_sent_wo_eos, input_sent_w_eos, original_cluster_str):
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

