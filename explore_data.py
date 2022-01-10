import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

filename = '../REL/data/wiki_2019/basic_data/wiki_name_id_map.txt'
target_word = "Nikola Tesla Satellite Award"

with open(filename) as openfile:
    for line in openfile:
        if target_word in line:
            print(line[:100])

filename = 'title2id.txt'

with open(filename) as openfile:
    for line in enumerate(openfile):
        str_line = str(line)
        starting_index = str_line.index(target_word)

        if starting_index == -1:
            print('Not found')
        else:
            print(str_line[starting_index-50: starting_index+50])

'''
filename = 'wikipedia_id2local_id.txt'
target_id = '61125864'#'5879583'

with open(filename) as openfile:
    for line in enumerate(openfile):
        str_line = str(line)
        starting_index = str_line.index(target_word)

        if starting_index == -1:
            print('Not found')
        else:
            print(str_line[starting_index-50: starting_index+50])
'''