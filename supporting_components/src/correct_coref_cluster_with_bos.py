import sys, os, ast
def correct_cluster_with_bos(input_sent_wo_bos, input_sent_w_bos, original_cluster_str):
    """
        Convert coref-clusters of sentences without _bos to coref-cluster of sentences with _bos
        Input:  a sentence without _bos
                a cluster of sentence without _bos
                a sentence with _bos
        Output: a cluster of sentence with _bos

    """
    list_word_wo_bos = input_sent_wo_bos.strip().split()
    list_word_w_bos = input_sent_w_bos.strip().split()
    
    dict_word_wo_bos_to_idx = {}

    dict_map = {}
    dict_idx = {}
    w_bos_idx = 0
    break_check = 0
    for idx, word in enumerate(list_word_wo_bos):
        while word != list_word_w_bos[w_bos_idx]:
            w_bos_idx+=1
            if break_check==10:
                print('something\'s wrong')
                sys.exit()

        
        dict_map[word+'-|-'+str(idx)] = word+'-|-'+str(w_bos_idx)
        dict_idx[idx] = w_bos_idx

    dict_cluster_bos = {}
    cluster_wo_bos = ast.literal_eval(original_cluster_str.strip())
    expected_clusters = []
    for cluster in cluster_wo_bos:
        temp_cluster = []
        for item in cluster:
            _temp_content = []
            for _content in item:
                _temp_content.append(dict_idx.get(_content))
            temp_cluster.append(_temp_content)
        expected_clusters.append(temp_cluster)
    
    return str(expected_clusters)

def add_bos_into_cluster(cluster):
    cluster_wo_bos = ast.literal_eval(cluster.strip())
    cluster_w_bos = []

    for clus in cluster_wo_bos:
        tmp_clus = []
        for item in clus:
            tmp_clus.append([item[0]+1, item[1]+1])
        cluster_w_bos.append(tmp_clus)
    return str(cluster_w_bos)