user_ids, item_ids = set(), set()

with open('rerank_data.txt', 'r') as fin:
    for line in fin:
        line = line.strip()
        strs = line.split('\t')
        user_ids.add(int(strs[0]))

        for x in strs[1].split(','):
            item_ids.add(int(x))
        for x in strs[2].split(','):
            item_ids.add(int(x))

print('user_ids len: ', len(user_ids))
print('item_ids len: ', len(item_ids))

with open('user_ids.txt', 'w') as fout:
    user_ids_list = sorted(list(user_ids))
    for x in user_ids_list:
        fout.write(str(x) + '\n')


with open('item_ids.txt', 'w') as fout:
    item_ids_list = sorted(list(item_ids))
    for x in item_ids_list:
        fout.write(str(x) + '\n')