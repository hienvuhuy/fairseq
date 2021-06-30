import sys, os, glob

# Note that:
# We have to run for full dataset first
# fairseq-preprocess --source-lang en --target-lang ru --trainpref /home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/lower/train --validpref /home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/lower/valid --testpref /home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/lower/test --destdir ./full_set --thresholdtgt 0 --thresholdsrc 0 --workers 20

if len(sys.argv) != 11:
    print("Error: Run "+sys.argv[0]+" with wrong arguments. Please check again!!!")
    # print(sys.argv)
    sys.exit()

DATA_PATH = sys.argv[1]
src_lang = sys.argv[2]
tgt_lang = sys.argv[3]
TEMP_PATH = sys.argv[4]
thresholdsrc = sys.argv[5]
thresholdtgt = sys.argv[6]
workers = sys.argv[7]
DESTINATION_PATH = sys.argv[8]
SPLITTED_FOLDER = sys.argv[9]
prefix=sys.argv[10]

if DATA_PATH[-1]!='/':
    DATA_PATH = DATA_PATH+'/'

if TEMP_PATH[-1]!='/':
    TEMP_PATH = TEMP_PATH+'/'

if DESTINATION_PATH[-1]!='/':
    DESTINATION_PATH = DESTINATION_PATH+'/'

if SPLITTED_FOLDER[-1]!='/':
    SPLITTED_FOLDER = SPLITTED_FOLDER+'/'


command = "fairseq-preprocess --source-lang {} --target-lang {} ".format(src_lang, tgt_lang)
command+= " --trainpref {}train --validpref {}valid --testpref {}test ".format(DATA_PATH,DATA_PATH, DATA_PATH)
command+= " --destdir {} --thresholdtgt {} --thresholdsrc {} --workers {}".format(TEMP_PATH, thresholdtgt, thresholdsrc, workers)

print("      Running: {}".format(command))
os.system(command)

print("      Finishing extract vocabs in all training set")

src_dict = "dict.{}.txt".format(src_lang)
tgt_dict = "dict.{}.txt".format(tgt_lang)
# sys.exit()
sub_directory = glob.glob(SPLITTED_FOLDER+"*/")

for _index, _folder in enumerate(sub_directory):
    command = "fairseq-preprocess --source-lang {} --target-lang {} ".format(src_lang, tgt_lang)
    command += "--trainpref {}train ".format(_folder)
    command += "--testpref {}test ".format(_folder)
    command += "--validpref {}valid ".format(_folder)
    command += "--thresholdtgt 0 --thresholdsrc 0 --workers 20 "
    command += "--srcdict {}{} ".format(TEMP_PATH, src_dict)
    command += "--tgtdict {}{} ".format(TEMP_PATH, tgt_dict)
    command += "--destdir {} ".format(DESTINATION_PATH+'data-bin/'+prefix+str(_index).zfill(2))
    print (command)
    os.system(command)
    print ("done command")
    # sys.exit()
print("Done!!!") 


sys.exit()



PATH="/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/train/"
DESTINATION_PATH="/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/data-bin/"
# DICT_PATH="/home/cl/huyhien-v/Workspace/MT/my_fairseq/fairseq/data-bin/voita.en-ru/"
DICT_PATH="/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/full_set/"
CHECKPOINTS_PATH = "/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/checkpoints/"
ARCH_MODEL = "transformer_wmt_en_de"
MAX_TOKEN = 500
WANDB_NAME = "voita-en-ru-bpe-split"

src_lang = 'en'
src_dict = "dict.en.txt"
tgt_lang = 'ru'
tgt_dict = "dict.ru.txt"
prefix='voita-en-ru'
sub_directory = glob.glob(PATH+"*/")

for _index, _folder in enumerate(sub_directory):
    command = "fairseq-preprocess --source-lang {} --target-lang {} ".format(src_lang, tgt_lang)
    command += "--trainpref {}train ".format(_folder)
    command += "--testpref {}test ".format(_folder)
    command += "--validpref {}valid ".format(_folder)
    command += "--thresholdtgt 0 --thresholdsrc 0 --workers 20 "
    command += "--srcdict {}{} ".format(DICT_PATH, src_dict)
    command += "--tgtdict {}{} ".format(DICT_PATH, tgt_dict)
    command += "--destdir {} ".format(DESTINATION_PATH+prefix+str(_index).zfill(2))
    print (command)
    os.system(command)
    print ("done command")
    # sys.exit()
print("Done!!!") 
sys.exit()
# fairseq-preprocess --source-lang en --target-lang ru --trainpref $TEXT/train --validpref $TEXT/valid 
# --testpref $TEXT/test --destdir data-bin/voita.en-ru --thresholdtgt 0 --thresholdsrc 0 --workers 20

train_cm = ""
# DATA_PATH='data-bin/data-bin1:data-bin/data-bin2:data-bin/data-bin3:data-bin/data-bin4'
DATA_PATH=''
data_bin_folders = glob.glob(DESTINATION_PATH+'*/')
data_bin_folders = [item.replace(DESTINATION_PATH,'')[:-1] for item in data_bin_folders]
for _item in data_bin_folders:
    DATA_PATH+=DESTINATION_PATH+_item+':'
# print(DATA_PATH[:-1])
DATA_PATH = DATA_PATH[:-1]

train_cm += "CUDA_VISIBLE_DEVICES=1 "
train_cm += "fairseq-train {} --arch {} --optimizer adam ".format(DATA_PATH, ARCH_MODEL)
train_cm += "--adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt "
train_cm += "--warmup-updates 4000 --dropout 0.1 --weight-decay 0.0001 "
train_cm += "--criterion label_smoothed_cross_entropy --label-smoothing 0.1 "
train_cm += "--max-tokens {} --keep-last-epochs 10 --eval-bleu ".format(MAX_TOKEN)
train_cm += "--eval-bleu-args '{\"beam\": 4, \"max_len_a\": 1.2, \"max_len_b\": 10}' "
train_cm += "--eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu "
train_cm += "--maximize-best-checkpoint-metric --save-dir {}voita.en-ru.new ".format(CHECKPOINTS_PATH)
train_cm += "--num-workers 4 --update-freq 8 "
train_cm += "--wandb-project {} --seed 1234".format(WANDB_NAME)


print (train_cm)
# fairseq-train data-bin/voita.en-ru --arch transformer_wmt_en_de --optimizer adam 
# --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt 
# --warmup-updates 4000 --dropout 0.1 --weight-decay 0.0001 
# --criterion label_smoothed_cross_entropy --label-smoothing 0.1 
# --max-tokens 4000 --keep-last-epochs 10 --eval-bleu 
# --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' 
# --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu 
# --maximize-best-checkpoint-metric --save-dir checkpoints/voita.en-ru.new 
# --num-workers 4 

# CUDA_VISIBLE_DEVICES=0 fairseq-train /home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru00:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru01:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru02:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru03:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru04:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru05:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru06:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru07:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru08:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru09:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru10:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru11:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru12:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru13:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru14:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru15:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru16:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru17:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru18:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru19:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru20:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru21:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru22:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru23:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru24:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru25:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru26:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru27:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru28:/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/data-bin/voita-en-ru29 --arch transformer_wmt_en_de --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.1 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 500 --keep-last-epochs 10 --eval-bleu --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --save-dir /home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-split/checkpoints/voita.en-ru.new --num-workers 4 --update-freq 8 --wandb-project voita-en-ru-split --seed 1234