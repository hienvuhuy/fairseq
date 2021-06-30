import sys, os, glob

# if len(sys.argv) != 11:
#     print("Error: Run "+sys.argv[0]+" with wrong arguments. Please check again!!!")
#     # print(sys.argv)
#     sys.exit()

DATA_PATH=''
ARCH_MODEL=sys.argv[1]
OPTIMIZER=sys.argv[2]


data_bin_folders = glob.glob(DESTINATION_PATH+'*/')
data_bin_folders = [item.replace(DESTINATION_PATH,'')[:-1] for item in data_bin_folders]
for _item in data_bin_folders:
    DATA_PATH+=DESTINATION_PATH+_item+':'
# print(DATA_PATH[:-1])
DATA_PATH = DATA_PATH[:-1]


train_cm = ""
train_cm += "CUDA_VISIBLE_DEVICES=0 "
train_cm += "fairseq-train {} --arch {} --optimizer {} ".format(DATA_PATH, ARCH_MODEL, OPTIMIZER)
train_cm += " --adam-betas '(0.9,0.98)' --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt "
train_cm += " --warmup-updates 4000 --dropout 0.1 --weight-decay 0.0001 "
train_cm += " --criterion label_smoothed_cross_entropy --label-smoothing 0.1 "
train_cm += " --max-tokens {} --keep-last-epochs 10 --eval-bleu ".format(MAX_TOKEN)
train_cm += " --eval-bleu-args '{\"beam\": 4, \"max_len_a\": 1.2, \"max_len_b\": 10}' "
train_cm += " --eval-bleu-detok moses --eval-bleu-remove-bpe --best-checkpoint-metric bleu "
train_cm += " --maximize-best-checkpoint-metric --save-dir {}voita.en-ru.new ".format(CHECKPOINTS_PATH)
train_cm += " --num-workers 4 --update-freq 8 "
train_cm += " --wandb-project {} --seed 1234".format(WANDB_NAME)

print(train_cm)