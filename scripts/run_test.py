"""
This script is used to plot the ROC curves for the pretrained ML models and the ModSecurity WAF.
"""

import os
import matplotlib.pyplot as plt
import toml
import sys
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models import PyModSecurity
from src.data_loader import DataLoader
from src.extractor import ModSecurityFeaturesExtractor
from src.utils.plotting import plot_roc

# Set to True if you want to train the models on the WAFAMOLE dataset
DS_WAFAMOLE = False

if  __name__ == '__main__':
    settings        = toml.load('config.toml')
    crs_dir         = settings['crs_dir']
    crs_ids_path    = settings['crs_ids_path']
    models_path     = settings['models_path'] if not DS_WAFAMOLE else settings['models_wafamole_path']
    dataset_path    = settings['dataset_path'] if not DS_WAFAMOLE else settings['dataset_wafamole_path']
    figures_path    = settings['figures_path']
    paranoia_levels = settings['params']['paranoia_levels']
    models          = settings['params']['models']
    other_models    = settings['params']['other_models']
    penalties       = settings['params']['penalties']
    fig, axs        = plt.subplots(1, 4)
    zoom_axs        = dict()
    
    # ----------------------
    # LOADING DATASET PHASE
    # ----------------------
    print('[INFO] Loading dataset...')
    
    loader = DataLoader(
        malicious_path  = os.path.join(
            dataset_path, 
            f'malicious_test.{"pkl" if DS_WAFAMOLE else "json"}'
        ),
        legitimate_path = os.path.join(
            dataset_path, 
            f'legitimate_test.{"pkl" if DS_WAFAMOLE else "json"}'
        )
    )   
    
    if DS_WAFAMOLE:
        test_data = loader.load_data_pkl()
    else:
        test_data = loader.load_data()
    
    # ---------------------
    # STARTING EXPERIMENTS
    # ---------------------
    for pl in paranoia_levels:
        print(f'[INFO] Extracting features for PL {pl}...')
        
        extractor = ModSecurityFeaturesExtractor(
            crs_ids_path = crs_ids_path,
            crs_path     = crs_dir,
            crs_pl       = pl
        )
    
        xts, yts = extractor.extract_features(test_data)
        
        for model_name in other_models:
            print(f'[INFO] Evaluating {model_name} model for PL {pl}...')
                        
            if model_name == 'rf':
                label_legend = 'RF'
                settings     = {'color': 'green'}
                model        = joblib.load(
                    os.path.join(
                        models_path, 
                        f'rf_pl{pl}.joblib'
                    ))
                y_scores     = model.predict_proba(xts)[:, 1]
                
            elif  model_name == 'modsec':
                label_legend = 'ModSec'
                settings     = {'color': 'red'}
                waf          = PyModSecurity(
                    rules_dir = crs_dir,
                    pl        = pl
                )
                y_scores = waf.predict(test_data['payload'])
                
            elif model_name == 'infsvm':
                label_legend  = f'SecSVM '
                settings      = {'color': 'magenta'}
                model         = joblib.load(
                    os.path.join(
                        models_path, 
                        f'inf_svm_pl{pl}_t0.5.joblib'
                    ))
                y_scores      = model.decision_function(xts)
                
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

        for model_name in models:
            print(f'[INFO] Evaluating {model_name} model for PL {pl}...')
                
            for penalty in penalties: 
                if model_name == 'svc':
                    label_legend  = f'SVM - $\ell_{penalty[1]}$'
                    settings      = {'color': 'blue', 'linestyle': 'solid' if penalty == 'l1' else '--'}
                    model         = joblib.load(
                        os.path.join(
                            models_path, 
                            f'linear_svc_pl{pl}_{penalty}.joblib'
                        ))
                    y_scores     = model.decision_function(xts).reshape(-1,1)

                elif model_name == 'log_reg':
                    label_legend  = f'LR - $\ell_{penalty[1]}$'
                    settings      = {'color': 'orange', 'linestyle': 'solid' if penalty == 'l1' else '--'}
                    model         = joblib.load(
                        os.path.join(
                            models_path, 
                            f'log_reg_pl{pl}_{penalty}.joblib'
                        ))
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
        os.path.join(
            figures_path, 
            f'roc_curves{"_ds_wafamole" if DS_WAFAMOLE else ""}.pdf'
        ),
        dpi         = 600,
        format      = 'pdf',
        bbox_inches = "tight"
    )