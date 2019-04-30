import random


data_all = []

with open('rerank_data_10c.txt', 'r') as fin:
    for line in fin:
        strs = line.split('\t')
        if (len(strs) == 3) and (len(strs[1].split(',')) == 10) and (len(strs[2].split(',')) == 50):
            data_all.append(line.strip())


print('data_all size: ', len(data_all))

train_size = int(len(data_all) * 0.8)
test_size = len(data_all) - train_size

print('train_size: ', train_size)
print('test_size: ', test_size)

random.shuffle(data_all)
data_train = data_all[0 : train_size]
data_test = data_all[train_size : -1]


with open('rerank_data_10c_train.txt', 'w') as fout:
    for line in data_train:
        fout.write(line + '\n')


with open('rerank_data_10c_test.txt', 'w') as fout:
    for line in data_test:
        fout.write(line + '\n')