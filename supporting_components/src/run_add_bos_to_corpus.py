import sys, os 
from correct_coref_cluster_with_bos import add_bos_into_cluster
path_to_data = "/home/is/huyhien-v/Data/Raw/MT/En-Ru/voita_19/context_aware_with_target/sample_check/relevant"
list_text_files = ['en_train', 'en_test', 'en_dev', 'ru_dev', 'ru_test','ru_train']
list_coref_files = ["en_train_coref", "en_test_coref", "en_dev_coref"]

path_to_data = os.path.abspath(path_to_data)
print("  Add bos:")
# for _file in list_text_files:
#     # add bos
#     # command = """awk '{print "_bos ", $0} {path_to_data}/{file_in} > {path_to_data}/{file_in}.bos'""".format(path_to_data=path_to_data, file_in=_file)
#     command = "awk '{print \"_bos\",$0}' "
#     command += "{path_to_data}/{file_in} > {path_to_data}/{file_in}.bos".format(path_to_data=path_to_data, file_in=_file)
#     # print(command)
#     # sys.exit()
#     os.system(command)
#     # sys.exit()
#     # remove old files
#     command = "rm -rf {path_to_data}/{file_in}".format(path_to_data=path_to_data, file_in=_file)
#     os.system(command)

#     command = "mv {path_to_data}/{file_in}.bos {path_to_data}/{file_in}".format(path_to_data=path_to_data, file_in=_file)
#     os.system(command)

# sys.exit()
print("  Correct coref")
# correct clusters infor for _bos 

for _file in list_coref_files:
    file_in = path_to_data+'/'+_file
    list_lines = [i.strip() for i in open(file_in, 'r').readlines()]
    write_file = open(file_in+'.bos', 'w')

    for line in list_lines:
        write_file.write(add_bos_into_cluster(line).strip()+'\n')
    write_file.close()
    os.system('rm -rf {file_in}'.format(file_in=file_in))
    os.system('mv {file_in} {file_out}'.format(file_in=file_in+'.bos', file_out=file_in))

print("Done")

