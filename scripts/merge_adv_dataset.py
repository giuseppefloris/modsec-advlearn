import json
import os
import random


dataset1_path = 'data/dataset/malicious_test.json'
output_dir = 'wafamole_results/adv_payloads_test'
output_files = [
    "adv_train_test_inf_svm_pl4_rs20_100rounds.json",
    "adv_train_test_log_reg_pl4_rs20_100rounds.json",
    "adv_train_test_rf_pl4_rs20_100rounds.json",
    "adv_train_test_svm_linear_pl4_rs20_100rounds.json"
]

with open(dataset1_path, 'r') as f:
    dataset1 = json.load(f)

for output_file in output_files:

    with open(os.path.join(output_dir, output_file), 'r') as f:
        dataset2 = json.load(f)

    print(f"Processing {output_file}...")
    print("Length of dataset2:", len(dataset2))
    print("Length of dataset1:", len(dataset1))

    dataset1_copy = dataset1.copy()
    dataset1_copy[2000:4000] = dataset2
    
    merged_dataset = dataset1_copy
    #random.shuffle(merged_dataset)
    print("Length of merged dataset:", len(merged_dataset))

    output_path = os.path.join('data/dataset', f'{output_file}')
    with open(output_path, 'w') as f:
        json.dump(merged_dataset, f, indent=2, separators=(',', ': '))

    print(f"Saved merged dataset to {output_path}")

print("All datasets merged and saved successfully.")
