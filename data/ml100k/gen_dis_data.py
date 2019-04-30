import random


with open('rerank_data_test.txt', 'r') as fin:
    with open('dis_data_test.txt', 'w') as fout:
        for line in fin:
            strs = line.strip().split('\t')
            user = strs[0]
            card_pos = strs[1].split(',')
            item_cand = strs[2].split(',')
            item_pos = card_pos[0]

            item_cand_neg = set(item_cand)
            item_cand_neg.remove(item_pos)
            item_cand_neg = list(item_cand_neg)

            card_neg = []
            if random.random() < 0.3:
                while True:
                    card_neg.append(item_pos)
                    card_neg.extend(random.sample(item_cand_neg, k=3))
                    if ','.join(sorted(card_neg)) != ','.join(sorted(card_pos)):
                        break
                    else:
                        card_neg = []
            else:
                card_neg.extend(random.sample(item_cand_neg, k=4))

            fout.write(user + '\t' + ','.join(card_pos) + '\t' + '1' + '\n')
            fout.write(user + '\t' + ','.join(card_neg) + '\t' + '0' + '\n')
            fout.flush()