card_set = set()

with open('dis_data_10c_train.txt', 'r') as fin:
    for line in fin:
        strs = line.split('\t')
        card = ','.join([str(y) for y in sorted([int(x) for x in strs[1].split(',')])])
        card_set.add(card)

with open('dis_data_10c_test.txt', 'r') as fin:
    for line in fin:
        strs = line.split('\t')
        card = ','.join([str(y) for y in sorted([int(x) for x in strs[1].split(',')])])
        card_set.add(card)

print(len(card_set))