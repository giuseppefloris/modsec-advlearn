import json
import os


input_files = [
    "wafamole_results/results_dataset_2/adv_examples_retrained_test/output_svm_linear_l1_pl4_rs20_100rounds.json",
    "wafamole_results/results_dataset_2/adv_examples_retrained_test/output_svm_linear_l2_pl4_rs20_100rounds.json"
]

def clean_payload(payload):

    return payload.replace("\u00a0", " ")

def process_file(input_file):

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    

    adv_payloads = [clean_payload(entry['adv_payload']) for entry in data]
    print(len(adv_payloads))
    
    

    output_data = adv_payloads
    

    output_file = input_file.replace("output_", "adv_train_test_")
    

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {input_file} and saved to {output_file}")

if __name__ == "__main__":
    for file in input_files:
        process_file(file)
