"""
This script is used to train the models with different paranoia levels and penalties.
The trained models are saved as joblib files in the models directory.
"""

import toml
import os
import sys
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.extractor import ModSecurityFeaturesExtractor
from src.models import InfSVM
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


if __name__  == '__main__':
    settings         = toml.load('config.toml')
    crs_dir          = settings['crs_dir']
    crs_ids_path     = settings['crs_ids_path']
    models_path      = settings['models_path']
    models_path_ds2  = settings['models_path_ds2']
    figures_path     = settings['figures_path']
    dataset_path     = settings['dataset_path']
    dataset2_path    = settings['dataset2_path']
    paranoia_levels  = settings['params']['paranoia_levels']
    models           = list(filter(lambda model: model != 'modsec', settings['params']['other_models']))
    models           +=settings['params']['models']
    penalties        = settings['params']['penalties']
    t                = [0.5, 1]
    
    # LOADING DATASET PHASE
    print('[INFO] Loading dataset...')
    
    # Dataset ModSec-learn
    loader = DataLoader(
        malicious_path  = os.path.join(dataset_path, 'malicious_train.json'),
        legitimate_path = os.path.join(dataset_path, 'legitimate_train.json')
    )    
    training_data = loader.load_data()

    # Dataset WAF-A-MoLE
    loader = DataLoader(
        malicious_path  = os.path.join(dataset2_path, 'sqli_train.pkl'),
        legitimate_path = os.path.join(dataset2_path, 'benign_train.pkl')
    )    
    training_data2 = loader.load_data_pkl()

    models_weights = dict()
    
    for pl in paranoia_levels:
        # FEATURE EXTRACTION PHASE
        print('[INFO] Extracting features for PL {}...'.format(pl))
        
        extractor = ModSecurityFeaturesExtractor(
            crs_ids_path = crs_ids_path,
            crs_path     = crs_dir,
            crs_pl       = pl
        )
    
        xtr, ytr = extractor.extract_features(training_data2)

        # TRAINING PHASE
        for model_name in models:
            print('[INFO] Training {} model for PL {}...'.format(model_name, pl))
            
            if model_name == 'infsvm':
                for numbers in t: 
                    model = InfSVM(numbers)
                    model.fit(xtr, ytr)
                    joblib.dump(
                        model, 
                        os.path.join(models_path_ds2, 'inf_svm_pl{}_t{}.joblib'.format(pl,numbers))
                    )
                    
            if model_name == 'svc':
                for penalty in penalties:
                    model = LinearSVC(
                        C             = 0.5,
                        penalty       = penalty,
                        dual          = 'auto',
                        class_weight  = 'balanced',
                        random_state  = 77,
                        fit_intercept = False,
                    )
                    model.fit(xtr, ytr)
                    joblib.dump(
                        model, 
                        os.path.join(models_path_ds2, 'linear_svc_pl{}_{}.joblib'.format(pl, penalty))
                    )
                        
            elif model_name == 'rf':
                model = RandomForestClassifier(
                    class_weight = 'balanced',
                    random_state = 77,
                    n_jobs       = -1
                )
                model.fit(xtr, ytr)
                joblib.dump(
                    model, 
                    os.path.join(models_path_ds2, 'rf_pl{}.joblib'.format(pl))
                )

            elif model_name == 'log_reg':
                for penalty in penalties:
                    model = LogisticRegression(
                        C            = 0.5,
                        penalty      = penalty,
                        class_weight = 'balanced',
                        random_state = 77,
                        n_jobs       = -1,
                        max_iter     = 1000,
                        solver       = 'saga'
                    )
                    model.fit(xtr, ytr)
                    joblib.dump(
                        model, 
                        os.path.join(models_path_ds2, 'log_reg_pl{}_{}.joblib'.format(pl, penalty))
                    )