"""
This script is used to plot the ROC curves for the models evaluated against the adversarial dataset.
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
        
    # Preparing path templates for the adversarial dataset
    adv_infsvm_pl_path = os.path.join(
        dataset_path, 
        f'adv_test_inf_svm_pl%s_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )   
    ms_adv_pl_path = os.path.join(
        dataset_path, 
        f'adv_test_ms_pl%s_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_log_reg_l1_path = os.path.join(
        dataset_path, 
        f'adv_test_log_reg_l1_pl%s_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_log_reg_l2_path  = os.path.join( 
        dataset_path, 
        f'adv_test_log_reg_l2_pl%s_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_svm_linear_l1_path = os.path.join(
        dataset_path, 
        f'adv_test_svm_linear_l1_pl%s_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_svm_linear_l2_path = os.path.join(
        dataset_path, 
        f'adv_test_svm_linear_l2_pl%s_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    adv_rf_path = os.path.join(
        dataset_path, 
        f'adv_test_rf_pl%s_rs20_100rounds.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    legitimate_test_path = os.path.join(
        dataset_path, 
        f'legitimate_test.{"pkl" if DS_WAFAMOLE else "json"}'
    )
    
    print('[INFO] Loading dataset...')

    loader = DataLoader(
        malicious_path  = os.path.join(
            dataset_path, 
            f'malicious_test.{"pkl" if DS_WAFAMOLE else "json"}'
        ),
        legitimate_path = legitimate_test_path
    ) 

    if DS_WAFAMOLE:
        test_data = loader.load_data_pkl()
    else:
        test_data = loader.load_data()
    
    # ---------------------
    # STARTING EXPERIMENTS
    # ---------------------
    for pl in paranoia_levels:
        # Loading the adversarial datasets
        adv_ms_loader = DataLoader(
            malicious_path  = ms_adv_pl_path % pl,
            legitimate_path = legitimate_test_path
        )
        adv_inf_svm_loader = DataLoader(
            malicious_path  = adv_infsvm_pl_path % pl,
            legitimate_path = legitimate_test_path
        )
        adv_log_reg_l1_loader = DataLoader(
            malicious_path  = adv_log_reg_l1_path % pl,
            legitimate_path = legitimate_test_path
        )
        adv_log_reg_l2_loader = DataLoader(
            malicious_path  = adv_log_reg_l2_path % pl,
            legitimate_path = legitimate_test_path
        )
        adv_svm_linear_l1_loader = DataLoader(
            malicious_path  = adv_svm_linear_l1_path % pl,
            legitimate_path = legitimate_test_path
        )
        adv_svm_linear_l2_loader = DataLoader(
            malicious_path  = adv_svm_linear_l2_path % pl,
            legitimate_path = legitimate_test_path
        )
        adv_rf_loader = DataLoader(
            malicious_path  = adv_rf_path % pl,
            legitimate_path = legitimate_test_path
        )

        if DS_WAFAMOLE:
            adv_test_ms_data            = adv_ms_loader.load_data_pkl()
            adv_test_inf_svm_data       = adv_inf_svm_loader.load_data_pkl()
            adv_test_log_reg_l1_data    = adv_log_reg_l1_loader.load_data_pkl()
            adv_test_log_reg_l2_data    = adv_log_reg_l2_loader.load_data_pkl()
            adv_test_svm_linear_l1_data = adv_svm_linear_l1_loader.load_data_pkl()
            adv_test_svm_linear_l2_data = adv_svm_linear_l2_loader.load_data_pkl()
            adv_test_rf_data            = adv_rf_loader.load_data_pkl()
        else:
            adv_test_ms_data            = adv_ms_loader.load_data()
            adv_test_inf_svm_data       = adv_inf_svm_loader.load_data()
            adv_test_log_reg_l1_data    = adv_log_reg_l1_loader.load_data()
            adv_test_log_reg_l2_data    = adv_log_reg_l2_loader.load_data()
            adv_test_svm_linear_l1_data = adv_svm_linear_l1_loader.load_data()
            adv_test_svm_linear_l2_data = adv_svm_linear_l2_loader.load_data()
            adv_test_rf_data            = adv_rf_loader.load_data()
            
        print(f'[INFO] Extracting features for PL {pl}...')
        
        extractor = ModSecurityFeaturesExtractor(
            crs_ids_path = crs_ids_path,
            crs_path     = crs_dir,
            crs_pl       = pl
        )

        xts                  , yts                   = extractor.extract_features(test_data)
        adv_ms_xts           , adv_ms_yts            = extractor.extract_features(adv_test_ms_data)
        adv_inf_svm_xts      , adv_inf_svm_yts       = extractor.extract_features(adv_test_inf_svm_data)
        adv_log_reg_l1_xts   , adv_log_reg_l1_yts    = extractor.extract_features(adv_test_log_reg_l1_data)
        adv_log_reg_l2_xts   , adv_log_reg_l2_yts    = extractor.extract_features(adv_test_log_reg_l2_data)
        adv_svm_linear_l1_xts, adv_svm_linear_l1_yts = extractor.extract_features(adv_test_svm_linear_l1_data)
        adv_svm_linear_l2_xts, adv_svm_linear_l2_yts = extractor.extract_features(adv_test_svm_linear_l2_data)
        adv_rf_xts           , adv_rf_yts            = extractor.extract_features(adv_test_rf_data)
        
        for model_name in other_models:
            print(f'[INFO] Evaluating {model_name} model for PL {pl}...')
                
            if model_name == 'modsec':
                label_legend = 'ModSec'
                settings     = {'color': 'red'}
                adv_settings = {'color': 'red', 'linestyle': 'dashed'}
                waf          = PyModSecurity(
                    rules_dir = crs_dir,
                    pl        = pl
                )
                y_scores        = waf.predict(test_data['payload'])
                ms_adv_y_scores = waf.predict(adv_test_ms_data['payload'])
                
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
                    pl                 = pl
                )       
                plot_roc(
                    adv_ms_yts,
                    ms_adv_y_scores,
                    ax                 = axs.flatten()[pl-1],
                    settings           = adv_settings,
                    plot_rand_guessing = False,
                    log_scale          = True,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False,
                    pl                 = pl
                )
  
            elif model_name == 'infsvm':
                label_legend = f'SecSVM'
                settings     = {'color': 'darkmagenta'}
                adv_settings = {'color': 'darkmagenta', 'linestyle': 'dashed'}
                model        = joblib.load(
                    os.path.join(models_path, 'inf_svm_pl{}_t0.5.joblib'.format(pl))
                )
                y_scores     = model.decision_function(xts)
                adv_y_scores = model.decision_function(adv_inf_svm_xts)
                
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
                    pl                 = pl
                )                 
                plot_roc(
                    adv_inf_svm_yts,
                    adv_y_scores,
                    ax                 = axs.flatten()[pl-1],
                    settings           = adv_settings,
                    plot_rand_guessing = False,
                    log_scale          = True,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False,
                    pl                 = pl
                )

            elif model_name == 'rf':
                label_legend = 'RF'
                settings     = {'color': 'green'}
                adv_settings = {'color': 'green', 'linestyle': 'dashed'}
                model        = joblib.load(
                    os.path.join(models_path, 'rf_pl{}.joblib'.format(pl))
                )
                y_scores     = model.predict_proba(xts)[:, 1]
                adv_y_scores = model.predict_proba(adv_rf_xts)[:, 1]
                
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
                    pl                 = pl
                )
                plot_roc(
                    adv_rf_yts,
                    adv_y_scores,
                    ax                 = axs.flatten()[pl-1],
                    settings           = adv_settings,
                    plot_rand_guessing = False,
                    log_scale          = True,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False,
                    pl                 = pl
                )
                    
        for model_name in models:
            print('[INFO] Evaluating {} model for PL {}...'.format(model_name, pl))
            for penalty in penalties:   
                if model_name == 'svc':
                    label_legend = f'SVM – $\ell_{penalty[1]}$'
                    settings     = {'color': 'blue' if penalty == 'l1' else 'aqua', 'linestyle': 'solid'}
                    adv_settings = {'color': 'blue' if penalty == 'l1' else 'aqua', 'linestyle': 'dashed'}
                    model        = joblib.load(
                        os.path.join(models_path, 'linear_svc_pl{}_{}.joblib'.format(pl, penalty))
                    )
                    y_scores     = model.decision_function(xts)
                    adv_y_scores = model.decision_function(adv_svm_linear_l1_xts if penalty == 'l1' else adv_svm_linear_l2_xts)
                    
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
                        pl                 = pl
                    )   
                    plot_roc(
                        adv_svm_linear_l1_yts if penalty == 'l1' else adv_svm_linear_l2_yts,
                        adv_y_scores,
                        ax                 = axs.flatten()[pl-1],
                        settings           = adv_settings,
                        plot_rand_guessing = False,
                        log_scale          = True,
                        update_roc_values  = True if pl == 1 else False,
                        include_zoom       = False,
                        pl                 = pl
                    )
                    
                elif model_name == 'log_reg':
                    label_legend = f'LR – $\ell_{penalty[1]}$'
                    settings     = {'color': 'orange' if penalty == 'l1' else 'chocolate', 'linestyle': 'solid'}
                    adv_settings = {'color': 'orange' if penalty =='l1' else 'chocolate', 'linestyle': 'dashed'}
                    model        = joblib.load(
                        os.path.join(models_path, 'log_reg_pl{}_{}.joblib'.format(pl, penalty))
                    )
                    y_scores     = model.predict_proba(xts)[:, 1]
                    adv_y_scores = model.predict_proba(adv_log_reg_l1_xts if penalty == 'l1' else adv_log_reg_l2_xts)[:, 1]
                    
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
                        pl                 = pl
                    )   
                    plot_roc(
                        adv_log_reg_l1_yts if penalty == 'l1' else adv_log_reg_l2_yts,
                        adv_y_scores,
                        ax                 = axs.flatten()[pl-1],
                        settings           = adv_settings,
                        plot_rand_guessing = False,
                        log_scale          = True,
                        update_roc_values  = True if pl == 1 else False,
                        include_zoom       = False,
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
        ncol           = 7,
        fontsize       = 13
    )
    fig.set_size_inches(17, 5)
    fig.tight_layout(pad = 2.0)
    fig.savefig(
        os.path.join(
            figures_path, 
            f'roc_curves_test_adv{"_ds_wafamole" if DS_WAFAMOLE else ""}.pdf'
        ),
        dpi         = 600,
        format      = 'pdf',
        bbox_inches = "tight"
    )