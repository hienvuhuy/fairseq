# tag coref voita 
#   chia thanh cac part
#   usage: python tag_coref_voita_wo_tokenizer.py 
#           CUDA_DEVICE_NUM     PART
#   PART: 1-4
#       Part=-1: all data

import sys, os, re
from unicodedata import name

# from allennlp.common.tqdm import Tqdm
from allennlp.common.util import lazy_groups_of

from allennlp_models import pretrained
from allennlp.predictors.predictor import Predictor
# from tqdm import tqdm
import tqdm


parts = 5
stdstring = lambda x: ' '.join(x.strip().split())
find_x_in_y = lambda x, y: [m.start for m in re.finditer(x, y)]
remove_eos = lambda x: x.replace('_eos', '')


def split(length, n_parts, selected):
    # return (first index, end index)
    # length = len(data)
    if selected == 0: # dem tu 1
        selected = 1
    distance = length//n_parts
    if selected == n_parts:
        return (distance*(selected-1), length)
    return distance*(selected-1), distance*(selected)

def get_span_words(span, document):
    return ' '.join(document[span[0]:span[1]+1])

def print_clusters(prediction):
    document, clusters = prediction['document'], prediction['clusters']
    out_str = ''
    for cluster in clusters:
        out_str= get_span_words(cluster[0], document) + ': '
        out_str+=' '+"[{"+'; '.join([get_span_words(span, document) for span in cluster])+"}]"
        out_str+=' ; '
    return out_str
def gen_data_without_position_eos(list_of_sentences):
    # return sent ||| clusters info
    list_output = []
    data = []
    for sent in list_of_sentences:
        # data.append({'document':stdstring(remove_eos(sent))})
        data.append(sent.strip().split())
    # for batch in Tqdm.tqdm(lazy_groups_of(data, batch_size)):
    for sent in tqdm.tqdm(data):
        # from pudb import set_trace; set_trace()
        simple_out = predictor.predict_tokenized(sent)
        # predictor.predict_tokenized ?? why not?
        list_output.append(simple_out)

    return list_output

name_selected = ''
if len(sys.argv) > 2:
    cuda_device = int(sys.argv[1])
    selected = str(sys.argv[2])
    if int(selected) > 0:
        name_selected = '.part'+selected
    # if int(selected) < 0:
    #     selected = ''

print("CUDA_DEVICES = ", cuda_device)
print("part = ", name_selected)

spanbert_path ='/home/is/huyhien-v/Data/Pretrained-Models/AllenNLP/Coref/coref-spanbert-large-2021.03.10.tar.gz'


# file_in = '/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware_without_eos/en_train'
# file_out = "/home/cl/huyhien-v/Workspace/MT/my_fairseq/probing/fairseq-probing/supporting_components/output/coref/en_train_voita{part}.txt".format(part=str(name_selected))

file_in = '/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware_without_eos/en_test'
file_out = "/home/cl/huyhien-v/Workspace/MT/my_fairseq/probing/fairseq-probing/supporting_components/output/coref/en_test_voita{part}.txt".format(part=str(name_selected))

# file_in = '/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware_without_eos/en_dev'
# file_out = "/home/cl/huyhien-v/Workspace/MT/my_fairseq/probing/fairseq-probing/supporting_components/output/coref/en_dev_voita{part}.txt".format(part=str(name_selected))


listAllLine = [i.strip() for i in open(file_in, 'r').readlines()]
# from pudb import set_trace; set_trace()
if int(selected) > 0:
    bg, ed = split(len(listAllLine), parts, int(selected))
    listLine = listAllLine[bg:ed]
else:
    listLine = listAllLine




list_sents = [i.strip() for i in open(file_in, 'r').readlines()]


predictor = Predictor.from_path(spanbert_path, cuda_device=cuda_device)
data_sents = gen_data_without_position_eos(listLine)

outfile = open(file_out, 'w')
for item in data_sents:
    # from pudb import set_trace; set_trace()
    outfile.write(' '.join(item['document']) + ' ||| ' + str(item['clusters'])+'\n')
    # outfile.write(item+'\n')
outfile.close()
print('done')