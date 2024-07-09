"""
This script is used to plot the ROC curves for the pretrained ML models and the ModSecurity WAF.
"""

import os
import matplotlib.pyplot as plt
import toml
import sys
import joblib
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import PyModSecurity
from src.data_loader import DataLoader
from src.extractor import ModSecurityFeaturesExtractor
from src.utils.plotting import plot_roc


if  __name__ == '__main__':
    settings         = toml.load('config.toml')
    crs_dir          = settings['crs_dir']
    crs_ids_path     = settings['crs_ids_path']
    models_path      = settings['models_path']
    models_path_ds2  = settings['models_path_ds2']
    figures_path     = settings['figures_path']
    dataset_path     = settings['dataset_path']
    dataset2_path    = settings['dataset2_path']
    paranoia_levels  = settings['params']['paranoia_levels']
    models           = settings['params']['models']
    other_models     = settings['params']['other_models']
    penalties        = settings['params']['penalties']
    fig, axs         = plt.subplots(1, 4)
    zoom_axs         = dict()
    
    # LOADING DATASET PHASE
    print('[INFO] Loading dataset...')
    
    # Dataset1
    legitimate_train_path = os.path.join(dataset_path, 'legitimate_train.json')
    malicious_train_path  = os.path.join(dataset_path, 'malicious_train.json')
    legitimate_test_path  = os.path.join(dataset_path, 'legitimate_test.json')
    malicious_test_path   = os.path.join(dataset_path, 'malicious_test.json')
    
    loader = DataLoader(
        malicious_path  = malicious_train_path,
        legitimate_path = legitimate_train_path
    )    
    training_data = loader.load_data()

    loader = DataLoader(
        malicious_path  = malicious_test_path,
        legitimate_path = legitimate_test_path
    )    
    test_data = loader.load_data()
    
    # Dataset2
    benign_train_path = os.path.join(dataset2_path, 'benign_train.pkl')
    sqli_train_path   = os.path.join(dataset2_path, 'sqli_train.pkl')
    benign_test_path  = os.path.join(dataset2_path, 'benign_test.pkl')
    sqli_test_path    = os.path.join(dataset2_path, 'sqli_test.pkl')
    
    loader = DataLoader(
        malicious_path  = sqli_train_path,
        legitimate_path = benign_train_path
    )    
    training_data2 = loader.load_data_pkl()
    
    loader = DataLoader(
        malicious_path  = sqli_test_path,
        legitimate_path = benign_test_path
    )    
    test_data2 = loader.load_data_pkl()
    #print(test_data2)
    
    # STARTING EXPERIMENTS
    for pl in paranoia_levels:
        print('[INFO] Extracting features for PL {}...'.format(pl))
        
        extractor = ModSecurityFeaturesExtractor(
            crs_ids_path = crs_ids_path,
            crs_path     = crs_dir,
            crs_pl       = pl
        )
    
        xts, yts = extractor.extract_features(test_data2)
        

        for model_name in other_models:
            print('[INFO] Evaluating {} model for PL {}...'.format(model_name, pl))
                        
            if model_name == 'rf':
                label_legend = 'RF'
                color        = 'green'
                model        = joblib.load(
                    os.path.join(models_path_ds2, 'rf_pl{}.joblib'.format(pl))
                )
                y_scores     = model.predict_proba(xts)[:, 1]
                
            elif  model_name == 'modsec':
                label_legend = 'ModSec'
                color        = 'red'
                waf = PyModSecurity(
                    rules_dir = crs_dir,
                    pl        = pl
                )
                y_scores = waf.predict(test_data2['payload'])
                
            elif model_name == 'infsvm':
                label_legend  = f'SecSVM '
                color = 'magenta'
                model         = joblib.load(
                    os.path.join(models_path_ds2, 'inf_svm_pl{}_t0.5.joblib'.format(pl))
                )
                y_scores     = model.decision_function(xts)
                print("coef\n",model.coef_)
                print("intercept\n",model.intercept_)
                
            plot_roc(
                yts, 
                y_scores, 
                label_legend       = label_legend,
                ax                 = axs.flatten()[pl-1],
                settings           = {'color': color},
                plot_rand_guessing = False,
                log_scale          = True,
                update_roc_values  = True if pl == 1 else False,
                include_zoom       = False,
                zoom_axs           = zoom_axs,
                pl                 = pl
            )

        for model_name in models:
            print('[INFO] Evaluating {} model for PL {}...'.format(model_name, pl))
                
            for penalty in penalties: 
                
                if model_name == 'svc':
                    label_legend  = f'SVM - $\ell_{penalty[1]}$'
                    settings      = {'color': 'blue', 'linestyle': 'solid' if penalty == 'l1' else '--'}
                    model         = joblib.load(
                        os.path.join(models_path_ds2, 'linear_svc_pl{}_{}.joblib'.format(pl, penalty))
                    )
                    y_scores     = model.decision_function(xts).reshape(-1,1)

                elif model_name == 'log_reg':
                    label_legend  = f'LR - $\ell_{penalty[1]}$'
                    settings      = {'color': 'orange', 'linestyle': 'solid' if penalty == 'l1' else '--'}
                    model         = joblib.load(
                        os.path.join(models_path_ds2, 'log_reg_pl{}_{}.joblib'.format(pl, penalty))
                    )
                    y_scores      = model.predict_proba(xts)[:, 1]
                    
                plot_roc(
                    yts, 
                    y_scores, 
                    label_legend       = label_legend,
                    ax                 = axs.flatten()[pl-1],
                    settings           = settings,
                    plot_rand_guessing = False,
                    log_scale          = True,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False,
                    zoom_axs           = zoom_axs,
                    pl                 = pl
                )

    # Final global settings for the figure
            for idx, ax in enumerate(axs.flatten()):
                ax.set_title('PL {}'.format(idx+1), fontsize=16)
                ax.xaxis.set_tick_params(labelsize = 14)
                ax.yaxis.set_tick_params(labelsize = 14)
                ax.xaxis.label.set_size(16)
                ax.yaxis.label.set_size(16)

            handles, labels = axs.flatten()[0].get_legend_handles_labels()      
            
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
            fig.set_size_inches(17, 5)
            fig.tight_layout(pad = 2.0)
            fig.savefig(
                os.path.join(figures_path, 'roc_curves_ds2.pdf'),
                dpi         = 600,
                format      = 'pdf',
                bbox_inches = "tight"
            )

    
    