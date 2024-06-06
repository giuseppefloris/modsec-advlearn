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
from src.models import InfSVM2
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


if __name__        == '__main__':
    settings         = toml.load('config.toml')
    crs_dir          = settings['crs_dir']
    crs_ids_path     = settings['crs_ids_path']
    models_path      = settings['models_path']
    figures_path     = settings['figures_path']
    dataset_path     = settings['dataset_path']
    pl               = 4
    models           = list(filter(lambda model: model != 'modsec', settings['params']['other_models']))
    models           +=settings['params']['models']
    penalties        = settings['params']['penalties']
    t                = [0.5, 1]
    
    # LOADING DATASET PHASE
    print('[INFO] Loading dataset...')
    
    loader_adv_inf_svm = DataLoader(
        malicious_path  = os.path.join(dataset_path, 'adv_train_inf_svm_pl4_rs20_100rounds.json'),
        legitimate_path = os.path.join(dataset_path, 'legitimate_train.json')
    )
    loader_adv_log_reg = DataLoader(
        malicious_path  = os.path.join(dataset_path, 'adv_train_log_reg_pl4_rs20_100rounds.json'),
        legitimate_path = os.path.join(dataset_path, 'legitimate_train.json')
    )    
    loader_adv_rf = DataLoader(
        malicious_path=os.path.join(dataset_path, 'adv_train_rf_pl4_rs20_100rounds.json'),
        legitimate_path=os.path.join(dataset_path, 'legitimate_train.json')
    )
    loader_adv_svm = DataLoader(
        malicious_path=os.path.join(dataset_path, 'adv_train_svm_linear_pl4_rs20_100rounds.json'),
        legitimate_path=os.path.join(dataset_path, 'legitimate_train.json')
    )
    
    training_data_adv_inf_svm = loader_adv_inf_svm.load_data()
    training_data_adv_log_reg = loader_adv_log_reg.load_data()
    training_data_adv_rf = loader_adv_rf.load_data()
    training_data_adv_svm = loader_adv_svm.load_data()
    
    models_weights = dict()
    
    
    # FEATURE EXTRACTION PHASE
    print('[INFO] Extracting features for PL {}...'.format(pl))
    
    extractor = ModSecurityFeaturesExtractor(
        crs_ids_path = crs_ids_path,
        crs_path     = crs_dir,
        crs_pl       = pl
    )

    xtr_adv_inf_svm, ytr_adv_inf_svm = extractor.extract_features(training_data_adv_inf_svm)
    xtr_adv_log_reg, ytr_adv_log_reg = extractor.extract_features(training_data_adv_log_reg)
    xtr_adv_rf, ytr_adv_rf = extractor.extract_features(training_data_adv_rf)
    xtr_adv_svm, ytr_adv_svm = extractor.extract_features(training_data_adv_svm)
    # TRAINING PHASE
    for model_name in models:
        print('[INFO] Training {} model for PL {}...'.format(model_name, pl))
        
        if model_name == 'infsvm':
            for numbers in t: 
                model = InfSVM2(numbers)
                model.fit(xtr_adv_inf_svm, ytr_adv_inf_svm)
                joblib.dump(
                    model, 
                    os.path.join(models_path, 'adv_inf_svm_pl{}_t{}.joblib'.format(pl,numbers))
                )
                
        if model_name == 'svc':
            for penalty in penalties:
                    model = LinearSVC(
                        C             = 0.5,
                        penalty       = penalty,
                        dual='auto',
                        class_weight  = 'balanced',
                        random_state  = 77,
                        fit_intercept = False,
                    )
                    model.fit(xtr_adv_svm, ytr_adv_svm)
                    joblib.dump(
                        model, 
                        os.path.join(models_path, 'adv_linear_svc_pl{}_{}.joblib'.format(pl, penalty))
                    )
                    
        elif model_name == 'rf':
            model = RandomForestClassifier(
                class_weight = 'balanced',
                random_state = 77,
                n_jobs       = -1
            )
            model.fit(xtr_adv_rf, ytr_adv_rf)
            joblib.dump(
                model, 
                os.path.join(models_path, 'adv_rf_pl{}.joblib'.format(pl))
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
                model.fit(xtr_adv_log_reg, ytr_adv_log_reg)
                joblib.dump(
                    model, 
                    os.path.join(models_path, 'adv_log_reg_pl{}_{}.joblib'.format(pl, penalty))
                )