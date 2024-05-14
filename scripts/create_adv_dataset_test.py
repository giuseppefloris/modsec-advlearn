import json
import random


with open('data/dataset/malicious_test.json', 'r') as f:
    dataset1 = json.load(f)

with open('wafamole_results/adv_examples_test/output_ms_pl4_rs20_100rounds.json', 'r') as f:
    dataset2 = json.load(f)

print(len(dataset2))
print(len(dataset1))

dataset1[:2000] = dataset2

merged_dataset = dataset1
#random.shuffle(merged_dataset)

print(len(merged_dataset))

with open('adv_ds_ms_pl4.json', 'w') as f:
    json.dump(merged_dataset, f)

print("Datasets merged and shuffled successfully.")


