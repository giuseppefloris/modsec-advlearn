"""
This script is used to merge the adversarial payloads generated by the WAF-A-MoLE.
"""

import json
import os
import pickle

merge = True

if merge:
    dataset_path = 'data/dataset_wafamole/malicious_train.pkl'
    output_dir    = 'wafamole_results/results_dataset_wafamole/adv_payloads_train'
    output_files  = [
        "adv_train_inf_svm_pl4_rs20_100rounds.pkl",
        "adv_train_log_reg_l1_pl4_rs20_100rounds.pkl",
        "adv_train_log_reg_l2_pl4_rs20_100rounds.pkl",
        "adv_train_rf_pl4_rs20_100rounds.pkl",
        "adv_train_svm_linear_l1_pl4_rs20_100rounds.pkl",
        "adv_train_svm_linear_l2_pl4_rs20_100rounds.pkl",
    ]

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    for output_file in output_files:
        with open(os.path.join(output_dir, output_file), 'rb') as f:
            adv_dataset = pickle.load(f)

        print(f"Processing {output_file}...")
        print("Length of orginal dataset:", len(dataset))
        print("Length of adv samples:", len(adv_dataset))

        merged_dataset = dataset.copy()
        merged_dataset[0:len(adv_dataset)] = adv_dataset
        
        print("Length of merged dataset:", len(merged_dataset))
        
        output_path = os.path.join('data/dataset_wafamole', f'{output_file}')
        with open(output_path, 'wb') as f:
            pickle.dump(merged_dataset, f)

        print(f"Saved merged dataset to {output_path}")

    print("All datasets merged and saved successfully.")
else:
    json_files = [
        "wafamole_results/results_dataset_wafamole/adv_payloads_test/adv_train_test_svm_linear_l1_pl4_rs20_100rounds.json",
        "wafamole_results/results_dataset_wafamole/adv_payloads_test/adv_train_test_svm_linear_l2_pl4_rs20_100rounds.json"
    ]

    for json_file in json_files:
        with open(json_file, 'r') as file:
            data = json.load(file)
        
        pkl_file = json_file.replace('.json', '.pkl')
        with open(pkl_file, 'wb') as file:
            pickle.dump(data, file)

        print(f"Converted {json_file} to {pkl_file}")