import os
import toml
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.extractor import ModSecurityFeaturesExtractor


if __name__ == '__main__':
    settings      = toml.load('config.toml')
    dataset_path  = settings['dataset_path']
    crs_dir       = settings['crs_dir']
    crs_ids_path  = settings['crs_ids_path']

    # LOAD DATASET
    loader = DataLoader(
        malicious_path  = os.path.join(dataset_path, 'malicious/sqli'),
        legitimate_path = os.path.join(dataset_path, 'legitimate/legitimate')
    )    

    data = loader.load_data()   

    # EXTRACTS RULES IDS
    extractor = ModSecurityFeaturesExtractor(
        crs_ids_path = os.path.join(crs_ids_path, 'crs_sqli_ids_4.0.0.json'),
        crs_path     = crs_dir,
    )

    extractor.extract_crs_ids(data)