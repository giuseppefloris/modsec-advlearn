"""
This script is used to train the models with the adversarial dataset, different paranoia 
levels and penalties. The trained models are saved as joblib files in the models directory.
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
from sklearn.metrics import accuracy_score


if __name__        == '__main__':
    settings         = toml.load('config.toml')
    crs_dir          = settings['crs_dir']
    crs_ids_path     = settings['crs_ids_path']
    models_path      = settings['models_path']
    models_path_ds2  = settings['models_path_ds2']
    figures_path     = settings['figures_path']
    dataset_path     = settings['dataset_path']
    dataset2_path    = settings['dataset2_path']
    pl               = 4
    models           = list(filter(lambda model: model != 'modsec', settings['params']['other_models']))
    models           +=settings['params']['models']
    penalties        = settings['params']['penalties']
    t                = [0.5, 1]
    
    # LOADING DATASET PHASE
    print('[INFO] Loading dataset...')
    
    loader_adv_inf_svm = DataLoader(
        malicious_path  = os.path.join(dataset2_path, 'adv_train_inf_svm_pl4_rs20_100rounds.pkl'),
        legitimate_path = os.path.join(dataset2_path, 'benign_train.pkl')
    )
    loader_adv_log_reg_l1 = DataLoader(
        malicious_path  = os.path.join(dataset2_path, 'adv_train_log_reg_l1_pl4_rs20_100rounds.pkl'),
        legitimate_path = os.path.join(dataset2_path, 'benign_train.pkl')
    )    
    loader_adv_log_reg_l2 = DataLoader(
        malicious_path  = os.path.join(dataset2_path, 'adv_train_log_reg_l2_pl4_rs20_100rounds.pkl'),
        legitimate_path = os.path.join(dataset2_path, 'benign_train.pkl')
    )   
    loader_adv_rf = DataLoader(
        malicious_path  = os.path.join(dataset2_path, 'adv_train_rf_pl4_rs20_100rounds.pkl'),
        legitimate_path = os.path.join(dataset2_path, 'benign_train.pkl')
    )
    loader_adv_svm_l1 = DataLoader(                    
        malicious_path  = os.path.join(dataset2_path, 'adv_train_svm_linear_l1_pl4_rs20_100rounds.pkl'),
        legitimate_path = os.path.join(dataset2_path, 'benign_train.pkl')
    )
    loader_adv_svm_l2 = DataLoader(                    
        malicious_path  = os.path.join(dataset2_path, 'adv_train_svm_linear_l2_pl4_rs20_100rounds.pkl'),
        legitimate_path = os.path.join(dataset2_path, 'benign_train.pkl')
    )
    
    training_data_adv_inf_svm    = loader_adv_inf_svm.load_data_pkl()
    training_data_adv_log_reg_l1 = loader_adv_log_reg_l1.load_data_pkl()
    training_data_adv_log_reg_l2 = loader_adv_log_reg_l2.load_data_pkl()
    training_data_adv_rf         = loader_adv_rf.load_data_pkl()
    training_data_adv_svm_l1     = loader_adv_svm_l1.load_data_pkl()
    training_data_adv_svm_l2     = loader_adv_svm_l2.load_data_pkl()
    
    models_weights = dict()
    
    # FEATURE EXTRACTION PHASE
    print('[INFO] Extracting features for PL {}...'.format(pl))
    
    extractor = ModSecurityFeaturesExtractor(
        crs_ids_path = crs_ids_path,
        crs_path     = crs_dir,
        crs_pl       = pl
    )

    xtr_adv_inf_svm   , ytr_adv_inf_svm    = extractor.extract_features(training_data_adv_inf_svm)
    xtr_adv_log_reg_l1, ytr_adv_log_reg_l1 = extractor.extract_features(training_data_adv_log_reg_l1)
    xtr_adv_log_reg_l2, ytr_adv_log_reg_l2 = extractor.extract_features(training_data_adv_log_reg_l2)
    xtr_adv_rf        , ytr_adv_rf         = extractor.extract_features(training_data_adv_rf)
    xtr_adv_svm_l1    , ytr_adv_svm_l1     = extractor.extract_features(training_data_adv_svm_l1)
    xtr_adv_svm_l2    , ytr_adv_svm_l2     = extractor.extract_features(training_data_adv_svm_l2)
    
    # TRAINING PHASE
    for model_name in models:
        print('[INFO] Training {} model for PL {}...'.format(model_name, pl))
        
        if model_name == 'infsvm':
            for numbers in t: 
                model = InfSVM(numbers)
                model.fit(xtr_adv_inf_svm, ytr_adv_inf_svm)
                joblib.dump(
                    model, 
                    os.path.join(models_path_ds2, 'adv_inf_svm_pl{}_t{}.joblib'.format(pl,numbers))
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
                    max_iter      = 7000
                )
                
                if penalty == 'l1':
                    model.fit(xtr_adv_svm_l1, ytr_adv_svm_l1)
                    y_pred   = model.predict(xtr_adv_svm_l1)
                    accuracy = accuracy_score(ytr_adv_svm_l1, y_pred)
                else:
                    model.fit(xtr_adv_svm_l2, ytr_adv_svm_l2)
                
                joblib.dump(
                    model, 
                    os.path.join(models_path_ds2, 'adv_linear_svc_pl{}_{}.joblib'.format(pl,penalty))
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
                os.path.join(models_path_ds2, 'adv_rf_pl{}.joblib'.format(pl))
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
                
                if penalty == 'l1':
                    model.fit(xtr_adv_log_reg_l1, ytr_adv_log_reg_l1)
                else:
                    model.fit(xtr_adv_log_reg_l2, ytr_adv_log_reg_l2)
                
                joblib.dump(
                    model, 
                    os.path.join(models_path_ds2, 'adv_log_reg_pl{}_{}.joblib'.format(pl,penalty))
                )