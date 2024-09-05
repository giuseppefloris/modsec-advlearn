"""
This script is used to plot the ROC curves for the models trained with the adversarial dataset.
"""

import os
import matplotlib.pyplot as plt
import toml
import sys
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.extractor import ModSecurityFeaturesExtractor
from src.utils.plotting import plot_roc

# Set to True if you want to train the models on the WAFAMOLE dataset
DS_WAFAMOLE = True

if  __name__ == '__main__':    
    settings     = toml.load('config.toml')
    crs_dir      = settings['crs_dir']
    crs_ids_path = settings['crs_ids_path']
    models_path  = settings['models_path'] if not DS_WAFAMOLE else settings['models_wafamole_path']
    dataset_path = settings['dataset_path'] if not DS_WAFAMOLE else settings['dataset_wafamole_path']
    figures_path = settings['figures_path']
    models       = settings['params']['models']
    other_models = settings['params']['other_models']
    penalties    = settings['params']['penalties']
    pl           = 4
    fig, axs     = plt.subplots(1, 6)
    
    legitimate_test_path = os.path.join(
        dataset_path, 
        f'legitimate_test.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    malicious_test_path = os.path.join(
        dataset_path, 
        f'malicious_test.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    
    # Preparing paths for the adversarial datasets
    adv_infsvm_path_test = os.path.join(
        dataset_path, 
        f'adv_test_inf_svm_pl{pl}_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_log_reg_l1_path_test = os.path.join(
        dataset_path, 
        f'adv_test_log_reg_l1_pl{pl}_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_log_reg_l2_path_test = os.path.join(
        dataset_path, 
        f'adv_test_log_reg_l2_pl{pl}_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_svm_linear_l1_path_test = os.path.join(
        dataset_path, 
        f'adv_test_svm_linear_l1_pl{pl}_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_svm_linear_l2_path_test = os.path.join(
        dataset_path, 
        f'adv_test_svm_linear_l2_pl{pl}_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_rf_path_test = os.path.join(
        dataset_path, 
        f'adv_test_rf_pl{pl}_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    
    # Preparing path for the adversarial datasets generated using the adversarial trained models
    adv_infsvm_pl_path = os.path.join(
        dataset_path, 
        f'adv_train_test_inf_svm_pl{pl}_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_log_reg_l1_path = os.path.join(
        dataset_path, 
        f'adv_train_test_log_reg_l1_pl{pl}_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_log_reg_l2_path = os.path.join(
        dataset_path, 
        f'adv_train_test_log_reg_l2_pl{pl}_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_svm_linear_l1_path = os.path.join(
        dataset_path, 
        f'adv_train_test_svm_linear_l1_pl{pl}_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_svm_linear_l2_path = os.path.join(
        dataset_path, 
        f'adv_train_test_svm_linear_l2_pl{pl}_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_rf_path = os.path.join(
        dataset_path, 
        f'adv_train_test_rf_pl{pl}_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )

    print('[INFO] Loading dataset...')

    test_loader = DataLoader(
        malicious_path  = malicious_test_path,
        legitimate_path = legitimate_test_path
    )    
    adv_test_inf_svm_loader = DataLoader(
        malicious_path  = adv_infsvm_path_test,
        legitimate_path = legitimate_test_path
    )
    adv_test_log_reg_l1_loader = DataLoader(
        malicious_path  = adv_log_reg_l1_path_test,
        legitimate_path = legitimate_test_path
    )
    adv_test_log_reg_l2_loader = DataLoader(
        malicious_path  = adv_log_reg_l2_path_test,
        legitimate_path = legitimate_test_path
    )
    adv_test_svm_linear_l1_loader = DataLoader(
        malicious_path  = adv_svm_linear_l1_path_test,
        legitimate_path = legitimate_test_path
    )
    adv_test_svm_linear_l2_loader = DataLoader(
        malicious_path  = adv_svm_linear_l2_path_test,
        legitimate_path = legitimate_test_path
    )
    adv_test_rf_loader = DataLoader(
        malicious_path  = adv_rf_path_test,
        legitimate_path = legitimate_test_path
    )
    adv_train_test_inf_svm_loader = DataLoader(
        malicious_path  = adv_infsvm_pl_path,
        legitimate_path = legitimate_test_path
    )
    adv_train_test_log_reg_l1_loader = DataLoader(
        malicious_path  = adv_log_reg_l1_path,
        legitimate_path = legitimate_test_path
    )
    adv_train_test_log_reg_l2_loader = DataLoader(
        malicious_path  = adv_log_reg_l2_path,
        legitimate_path = legitimate_test_path
    )
    adv_train_test_svm_linear_l1_loader = DataLoader(
        malicious_path  = adv_svm_linear_l1_path,
        legitimate_path = legitimate_test_path
    )
    adv_train_test_svm_linear_l2_loader = DataLoader(
        malicious_path  = adv_svm_linear_l2_path,
        legitimate_path = legitimate_test_path
    )
    adv_train_test_rf_loader = DataLoader(
        malicious_path  = adv_rf_path,
        legitimate_path = legitimate_test_path
    )

    if DS_WAFAMOLE:
        test_data = test_loader.load_data_pkl()
        adv_test_inf_svm_data = adv_test_inf_svm_loader.load_data_pkl()
        adv_test_log_reg_l1_data = adv_test_log_reg_l1_loader.load_data_pkl()
        adv_test_log_reg_l2_data = adv_test_log_reg_l2_loader.load_data_pkl()
        adv_test_svm_linear_l1_data = adv_test_svm_linear_l1_loader.load_data_pkl()
        adv_test_svm_linear_l2_data = adv_test_svm_linear_l2_loader.load_data_pkl()
        adv_test_rf_data = adv_test_rf_loader.load_data_pkl()

        adv_train_test_inf_svm_data = adv_train_test_inf_svm_loader.load_data_pkl()
        adv_train_test_log_reg_l1_data = adv_train_test_log_reg_l1_loader.load_data_pkl()
        adv_train_test_log_reg_l2_data = adv_train_test_log_reg_l2_loader.load_data_pkl()
        adv_train_test_svm_linear_l1_data = adv_train_test_svm_linear_l1_loader.load_data_pkl()
        adv_train_test_svm_linear_l2_data = adv_train_test_svm_linear_l2_loader.load_data_pkl()
        adv_train_test_rf_data = adv_train_test_rf_loader.load_data_pkl()
    else:
        test_data                   = test_loader.load_data()
        adv_test_inf_svm_data       = adv_test_inf_svm_loader.load_data()
        adv_test_log_reg_l1_data    = adv_test_log_reg_l1_loader.load_data()
        adv_test_log_reg_l2_data    = adv_test_log_reg_l2_loader.load_data()
        adv_test_svm_linear_l1_data = adv_test_svm_linear_l1_loader.load_data()
        adv_test_svm_linear_l2_data = adv_test_svm_linear_l2_loader.load_data()
        adv_test_rf_data            = adv_test_rf_loader.load_data()
        
        adv_train_test_inf_svm_data       = adv_train_test_inf_svm_loader.load_data()
        adv_train_test_log_reg_l1_data    = adv_train_test_log_reg_l1_loader.load_data()
        adv_train_test_log_reg_l2_data    = adv_train_test_log_reg_l2_loader.load_data()
        adv_train_test_svm_linear_l1_data = adv_train_test_svm_linear_l1_loader.load_data()
        adv_train_test_svm_linear_l2_data = adv_train_test_svm_linear_l2_loader.load_data()
        adv_train_test_rf_data            = adv_train_test_rf_loader.load_data()

    print(f'[INFO] Extracting features for PL {pl}...')

    extractor = ModSecurityFeaturesExtractor(
        crs_ids_path = crs_ids_path,
        crs_path     = crs_dir,
        crs_pl       = pl
    )

    xts                   , yts                   = extractor.extract_features(test_data)
    adv_inf_svm_xts       , adv_inf_svm_yts       = extractor.extract_features(adv_test_inf_svm_data)
    adv_log_reg_l1_xts    , adv_log_reg_l1_yts    = extractor.extract_features(adv_test_log_reg_l1_data)
    adv_log_reg_l2_xts    , adv_log_reg_l2_yts    = extractor.extract_features(adv_test_log_reg_l2_data)
    adv_svm_linear_l1_xts , adv_svm_linear_l1_yts = extractor.extract_features(adv_test_svm_linear_l1_data)
    adv_svm_linear_l2_xts , adv_svm_linear_l2_yts = extractor.extract_features(adv_test_svm_linear_l2_data)
    adv_rf_xts            , adv_rf_yts            = extractor.extract_features(adv_test_rf_data)
        
    adv_train_inf_svm_xts      , adv_train_inf_svm_yts       = extractor.extract_features(adv_train_test_inf_svm_data)
    adv_train_log_reg_l1_xts   , adv_train_log_reg_l1_yts    = extractor.extract_features(adv_train_test_log_reg_l1_data)
    adv_train_log_reg_l2_xts   , adv_train_log_reg_l2_yts    = extractor.extract_features(adv_train_test_log_reg_l2_data)
    adv_train_svm_linear_l1_xts, adv_train_svm_linear_l1_yts = extractor.extract_features(adv_train_test_svm_linear_l1_data)
    adv_train_svm_linear_l2_xts, adv_train_svm_linear_l2_yts = extractor.extract_features(adv_train_test_svm_linear_l2_data)
    adv_train_rf_xts           , adv_train_rf_yts            = extractor.extract_features(adv_train_test_rf_data)
    
    # Evaluation phase
    model_settings = {
        'svc_l1': {
            'label'        : 'Linear SVM $\ell_1$',
            'color'        : 'orange',
            'adv_color'    : 'deepskyblue',
            'model'        : joblib.load(os.path.join(models_path, 'linear_svc_pl4_l1.joblib')),
            'adv_model'    : joblib.load(os.path.join(models_path, 'adv_linear_svc_pl4_l1.joblib')),
            'adv_xts'      : adv_svm_linear_l1_xts,
            'adv_yts'      : adv_svm_linear_l1_yts,
            'adv_train_xts': adv_train_svm_linear_l1_xts,
            'adv_train_yts': adv_train_svm_linear_l1_yts,
        },
        'svc_l2': {
            'label'        : 'Linear SVM $\ell_2$',
            'color'        : 'orange',
            'adv_color'    : 'deepskyblue',
            'model'        : joblib.load(os.path.join(models_path, 'linear_svc_pl4_l2.joblib')),
            'adv_model'    : joblib.load(os.path.join(models_path, 'adv_linear_svc_pl4_l2.joblib')),
            'adv_xts'      : adv_svm_linear_l2_xts,
            'adv_yts'      : adv_svm_linear_l2_yts,
            'adv_train_xts': adv_train_svm_linear_l2_xts,
            'adv_train_yts': adv_train_svm_linear_l2_yts
        },
        'rf': {
            'label'        : 'RF',
            'color'        : 'orange',
            'adv_color'    : 'deepskyblue',
            'model'        : joblib.load(os.path.join(models_path, 'rf_pl4.joblib')),
            'adv_model'    : joblib.load(os.path.join(models_path, 'adv_rf_pl4.joblib')),
            'adv_xts'      : adv_rf_xts,
            'adv_yts'      : adv_rf_yts,
            'adv_train_xts': adv_train_rf_xts,
            'adv_train_yts': adv_train_rf_yts
        },
        'log_reg_l1': {
            'label'        : 'LR $\ell_1$',
            'color'        : 'orange',
            'adv_color'    : 'deepskyblue',
            'model'        : joblib.load(os.path.join(models_path, 'log_reg_pl4_l1.joblib')),
            'adv_model'    : joblib.load(os.path.join(models_path, 'adv_log_reg_pl4_l1.joblib')),
            'adv_xts'      : adv_log_reg_l1_xts,
            'adv_yts'      : adv_log_reg_l1_yts,
            'adv_train_xts': adv_train_log_reg_l1_xts,
            'adv_train_yts': adv_train_log_reg_l1_yts
        },
        'log_reg_l2': {
            'label'        : 'LR $\ell_2$',
            'color'        : 'orange',
            'adv_color'    : 'deepskyblue',
            'model'        : joblib.load(os.path.join(models_path, 'log_reg_pl4_l2.joblib')),
            'adv_model'    : joblib.load(os.path.join(models_path, 'adv_log_reg_pl4_l2.joblib')),
            'adv_xts'      : adv_log_reg_l2_xts,
            'adv_yts'      : adv_log_reg_l2_yts,
            'adv_train_xts': adv_train_log_reg_l2_xts,
            'adv_train_yts': adv_train_log_reg_l2_yts
        },
        'infsvm': {
            'label'        : 'Sec SVM',
            'color'        : 'orange',
            'adv_color'    : 'deepskyblue',
            'model'        : joblib.load(os.path.join(models_path, 'inf_svm_pl4_t0.5.joblib')),
            'adv_xts'      : adv_inf_svm_xts,
            'adv_yts'      : adv_inf_svm_yts,
            'adv_train_xts': adv_train_inf_svm_xts,
            'adv_train_yts': adv_train_inf_svm_yts
        }
    }     
    
    for idx, (model_name, settings) in enumerate(model_settings.items()):
        ax = axs[idx]
        print(f'[INFO] Evaluating {model_name} model for PL 4...')
        
        # Evaluate the model against the test-set
        if model_name in ['svc_l1', 'svc_l2', 'infsvm']:
            y_scores = settings['model'].decision_function(xts) 
        else: 
            y_scores = settings['model'].predict_proba(xts)[:, 1]
        
        plot_roc(
            yts,
            y_scores,
            label_legend       = 'ModSec-Learn',
            ax                 = ax,
            settings           = {'color': settings['color'], 'linestyle': 'solid'},
            plot_rand_guessing = False,
            log_scale          = True,
            update_roc_values  = False,
            include_zoom       = False,
            pl                 = pl
        )
        
        # Evaluate the model against the adversarial test-set
        if model_name in ['svc_l1', 'svc_l2', 'infsvm']:
            adv_y_scores = settings['model'].decision_function(settings['adv_xts']) 
        else:
            adv_y_scores = settings['model'].predict_proba(settings['adv_xts'])[:, 1]
        
        plot_roc(
            settings['adv_yts'],
            adv_y_scores,
            ax                 = ax,
            settings           = {'color': settings['color'], 'linestyle': 'dashed'},
            plot_rand_guessing = False,
            log_scale          = True,
            update_roc_values  = False,
            include_zoom       = False,
            pl                 = pl
        )
        
        # Evaluate the adversarial trained model against the normal and adversarial test-set.
        # The Inf SVM is not considered here since it is not adversarially trained, it should
        # be already robust to adversarial examples.
        if model_name != 'infsvm':
            if model_name in ['svc_l1', 'svc_l2']:
                adv_train_y_scores = settings['adv_model'].decision_function(xts)
            else:
                adv_train_y_scores = settings['adv_model'].predict_proba(xts)[:, 1]
            
            plot_roc(
                yts,
                adv_train_y_scores,
                label_legend       = 'ModSec-AdvLearn',
                ax                 = ax,
                settings           = {'color': settings['adv_color'], 'linestyle': 'solid'},
                plot_rand_guessing = False,
                log_scale          = True,
                update_roc_values  = False,
                include_zoom       = False,
                pl                 = pl
            )
        
            if model_name in ['svc_l1', 'svc_l2']:
                adv_train_adv_y_scores = settings['adv_model'].decision_function(settings['adv_train_xts'])
            else:
                adv_train_adv_y_scores = settings['adv_model'].predict_proba(settings['adv_train_xts'])[:, 1]
            
            plot_roc(
                settings['adv_train_yts'],
                adv_train_adv_y_scores,
                ax                 = ax,
                settings           = {'color': settings['adv_color'], 'linestyle': 'dashed'},
                plot_rand_guessing = False,
                log_scale          = True,
                update_roc_values  = False,
                include_zoom       = False,
                pl                 = pl
            )

        # Final global settings for the figure
        ax.set_title(f'{settings["label"]} PL 4', fontsize = 16)
        ax.xaxis.set_tick_params(labelsize = 14)
        ax.yaxis.set_tick_params(labelsize = 14)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)

    handles, labels = axs[0].get_legend_handles_labels()      

    fig.legend(
        handles, 
        labels,
        loc            = 'upper center',
        bbox_to_anchor = (0.5, -0.01),
        fancybox       = True,
        shadow         = True,
        ncol           = 6,
        fontsize       = 13
    )
    fig.set_size_inches(22, 5)
    fig.tight_layout(pad = 2.0)
    fig.savefig(
        os.path.join(
            figures_path, 
            f'roc_curves_adv_train{"_ds_wafamole" if DS_WAFAMOLE else ""}.pdf'
        ),
        dpi         = 600,
        format      = 'pdf',
        bbox_inches = "tight"
    )
