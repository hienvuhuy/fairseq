import torch
import os, ast
NULL_ID_FOR_COREF = 0
def extract_clusters(gold_clusters):
    gold_clusters = [tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m) for gc in gold_clusters.tolist()]
    gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
    return gold_clusters

def extract_mentions_to_predicted_clusters_from_clusters(gold_clusters):
    mention_to_gold = {}
    for gc in gold_clusters:
        for mention in gc:
            mention_to_gold[tuple(mention)] = gc
    return mention_to_gold

def batch_extract_clusters(gold_clusters):
    # gold_clusters = [tuple(tuple(m) for m in gc if NULL_ID_FOR_COREF not in m) for gc in gold_clusters.tolist()]
    # gold_clusters = [cluster for cluster in gold_clusters if len(cluster) > 0]
    gold_clusters = [extract_clusters(cl) for cl in gold_clusters]
    return gold_clusters

def batch_extract_mentions_to_predicted_clusters_from_clusters(gold_clusters):
    return [extract_mentions_to_predicted_clusters_from_clusters(clusters) for clusters in gold_clusters]

#
def extract_clusters_for_decode(mention_to_antecedent):
    mention_to_antecedent = sorted(mention_to_antecedent)
    mention_to_cluster = {}
    clusters = []
    for mention, antecedent in mention_to_antecedent:
        if antecedent in mention_to_cluster:
            cluster_idx = mention_to_cluster[antecedent]
            clusters[cluster_idx].append(mention)
            mention_to_cluster[mention] = cluster_idx

        else:
            cluster_idx = len(clusters)
            mention_to_cluster[mention] = cluster_idx
            mention_to_cluster[antecedent] = cluster_idx
            clusters.append([antecedent, mention])
    clusters = [tuple(cluster) for cluster in clusters]
    return clusters, mention_to_cluster

padding_item = [[0,0]] # need to change?

def packed_data(sample_data, max_clusters, max_items_in_cluster, return_tensor=True):
    # max_cluster x max_item x 2
    sample_data = ast.literal_eval(sample_data)
    empty_cluster = padding_item*max_items_in_cluster

    # fill current list to match max_item
    x_padded = [row + padding_item * (max_items_in_cluster - len(row)) for row in sample_data]
    
    number_padded_cluster = max_clusters - len(sample_data)
    x_padded =  x_padded + [empty_cluster]*number_padded_cluster
    if return_tensor:

        return torch.tensor(x_padded)
    return x_padded

def packed_list_data(list_sample_data, max_clusters, max_items_in_cluster, return_tensor=True):
    list_output = []
    for i in list_sample_data:
        list_output.append(packed_data(i, max_clusters, max_items_in_cluster, return_tensor=True))
    if return_tensor:
        return torch.tensor(list_output)
    return list_output
