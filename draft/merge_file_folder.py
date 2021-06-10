import sys, os, shutil
import glob 

# Must have '/' in the end of path
DESTINATION_PATH = "/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/train/" 
NUMBER_OF_LINE_IN_FILE = '200000'

file_only = lambda x: x.strip().split('/')[-1]
name_only = lambda x: '.'.join(x.strip().split('.')[:-1])
# name_only = lambda x: x

input_source = "/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/original_full/train.en"
input_target = "/home/cl/huyhien-v/Workspace/MT/data/en-ru-baseline-bpe-split/original_full/train.ru"

len_input_source = os.popen('wc -l <'+input_source).read().strip()
len_input_target = os.popen('wc -l <'+input_target).read().strip()

assert len_input_source == len_input_target

print ("copying file....")
shutil.copy(input_source, DESTINATION_PATH+file_only(input_source))
shutil.copy(input_target, DESTINATION_PATH+file_only(input_target))

print ('\t.. done!')

print ('spliting file....')
os.system('split -l '+NUMBER_OF_LINE_IN_FILE+' '+DESTINATION_PATH+file_only(input_source)+' '+DESTINATION_PATH+file_only(input_source)+'.')
os.system('split -l '+NUMBER_OF_LINE_IN_FILE+' '+DESTINATION_PATH+file_only(input_target)+' '+DESTINATION_PATH+file_only(input_target)+'.')

print("\t.. done!")

print ('creating folder....')

list_source_input_files = glob.glob(DESTINATION_PATH+file_only(input_source)+'.*')
list_source_input_files = [file_only(x) for x in list_source_input_files]

list_target_input_files = glob.glob(DESTINATION_PATH+file_only(input_target)+'.*')
list_target_input_files = [file_only(x) for x in list_target_input_files]

SPLITTED_FOLDER = DESTINATION_PATH+'train_splitted/'
os.mkdir(SPLITTED_FOLDER)
for index, item in enumerate(list_source_input_files):
    # create folder
    os.mkdir(SPLITTED_FOLDER+"train"+str(index).zfill(2))
    
    shutil.move(DESTINATION_PATH+list_source_input_files[index], SPLITTED_FOLDER+"train"+str(index).zfill(2)+'/'+ name_only(list_source_input_files[index]))
    shutil.move(DESTINATION_PATH+list_target_input_files[index], SPLITTED_FOLDER+"train"+str(index).zfill(2)+'/'+ name_only(list_target_input_files[index]))

os.remove(DESTINATION_PATH+file_only(input_source))
os.remove(DESTINATION_PATH+file_only(input_target))
print("DONE!!!")
