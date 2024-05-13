import json

# models = ['inf_svm', 'linear_svc', 'log_reg', 'rf']
# paranoia_levels = [1, 2, 3, 4]
# base_path = "/models_newds"

# for model in models:
#     for level in paranoia_levels:
#         config = {
#             "sklearn_clf_path": f"{base_path}/{model}_pl{level}.joblib",
#             "crs_pl": level
#         }
#         with open(f"{model}_pl{level}_config.json", 'w') as file:
#             json.dump(config, file, indent=4)


import json
import os

# Directory paths
base_dir = "/home/gfloris/modsec-experiments/"
config_dir = os.path.join(base_dir,"modsec_config")
models_dir = os.path.join(base_dir, "data/models_newds")
rules_dir = os.path.join(base_dir, "coreruleset", "rules")
ids_path = os.path.join(base_dir, "data/crs_sqli_ids_4.0.0.json")

# Model types and their file extensions
model_types = {
    "inf_svm": "inf_svm_pl{pl}_t5.joblib",
    "linear_svc": "linear_svc_pl{pl}_l1.joblib",
    "log_reg": "log_reg_pl{pl}_l1.joblib",
    "rf": "rf_pl{pl}.joblib"
}

# PL values and threshold
pl_values = range(1, 5)
crs_threshold = 5

# Generate and save JSON files
for pl in pl_values:
    for model_name, model_filename in model_types.items():
        config = {
            "sklearn_clf_path": os.path.join(models_dir, model_filename.format(pl=pl)),
            "rules_path": rules_dir,
            "crs_rules_ids_path": ids_path,
            "crs_pl": pl,
            "crs_threshold": crs_threshold
        }
        
        # JSON file path
        json_file_path = os.path.join(config_dir, f"{model_name}_crs_pl{pl}_config.json")
        
        # Save JSON to a file
        with open(json_file_path, 'w') as json_file:
            json.dump(config, json_file, indent=4)
            
        print(f"Generated JSON file: {json_file_path}")
