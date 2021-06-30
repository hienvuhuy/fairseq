import sys, os, shutil, glob

# Argv1: source of the folder containing dev, test set. Ex: /home/abc/data
#   The source of folder have to contain: test.src, test.tgt, valid.src, valid.tgt
# Argv2: source language. Ex: en
# Argv3: target language. Ex: ru
# Argv4: destination path. Ex: /home/abc/experiments
# Argv6:

# print (len(sys.argv))
if len(sys.argv) != 5:
    print("Error: Run "+sys.argv[0]+" with wrong arguments. Please check again!!!")
    # print(sys.argv)
    sys.exit()

INPUT_PATH = sys.argv[1]
src_lang = sys.argv[2]
tgt_lang = sys.argv[3]
DESTINATION_PATH = sys.argv[4]

if INPUT_PATH[-1]!='/':
    INPUT_PATH = INPUT_PATH+'/'

if DESTINATION_PATH[-1]!='/':
    DESTINATION_PATH = DESTINATION_PATH + '/'

# INPUT_PATH = "/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/original_full/"
# DESTINATION_PATH = "/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/train/"
file_only = lambda x: x.strip().split('/')[-1]

list_files=['test.en', 'test.ru', 'valid.en', 'valid.ru']
list_files=[INPUT_PATH+item for item in list_files]

list_folders = glob.glob(DESTINATION_PATH+"*/")

for _file in list_files:
    for _folder in list_folders:
        # print ("copy from ", _file, _folder+file_only(_file))
        shutil.copy(_file, _folder+file_only(_file))

print('done')