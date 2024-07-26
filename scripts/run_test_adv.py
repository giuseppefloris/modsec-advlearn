"""
This script is used to plot the ROC curves for the models evaluated with the adversarial dataset.
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


if  __name__ == '__main__':
    settings         = toml.load('config.toml')
    crs_dir          = settings['crs_dir']
    crs_ids_path     = settings['crs_ids_path']
    models_path      = settings['models_path']
    figures_path     = settings['figures_path']
    dataset_path     = settings['dataset_path']
    paranoia_levels  = settings['params']['paranoia_levels'] 
    models           = settings['params']['models']
    other_models     = settings['params']['other_models']
    penalties        = settings['params']['penalties']
    fig, axs         = plt.subplots(1, 4)
    zoom_axs         = dict()
        
    # LOADING DATASET PHASE
    print('[INFO] Loading dataset...')

    legitimate_train_path = os.path.join(dataset_path, 'legitimate_train.json')
    malicious_train_path  = os.path.join(dataset_path, 'malicious_train.json')
    legitimate_test_path  = os.path.join(dataset_path, 'legitimate_test.json')
    malicious_test_path   = os.path.join(dataset_path, 'malicious_test.json')
    adv_infsvm_pl_paths   = [
        os.path.join(dataset_path, f'adv_test_inf_svm_pl{pl}_rs20_100rounds.json') for pl in range(1, 5)
    ]
    ms_adv_pl_paths       = [
        os.path.join(dataset_path, f'adv_test_ms_pl{pl}_rs20_100rounds.json') for pl in range(1, 5)
    ]
    adv_log_reg_l1_paths  = [ 
        os.path.join(dataset_path, f'adv_test_log_reg_pl{pl}_rs20_100rounds.json') for pl in range(1, 5)
    ]
    adv_log_reg_l2_paths  = [ 
        os.path.join(dataset_path, f'adv_test_log_reg_l2_pl{pl}_rs20_100rounds.json') for pl in range(1, 5)
    ]
    adv_svm_linear_l1_paths = [ 
        os.path.join(dataset_path, f'adv_test_svm_linear_pl{pl}_rs20_100rounds.json') for pl in range(1, 5)
    ]
    adv_svm_linear_l2_paths = [ 
        os.path.join(dataset_path, f'adv_test_svm_linear_l2_pl{pl}_rs20_100rounds.json') for pl in range(1, 5)
    ]
    adv_rf_paths = [
        os.path.join(dataset_path, f'adv_test_rf_pl{pl}_rs20_100rounds.json') for pl in range(1, 5)
    ]
    
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
    
    # STARTING EXPERIMENTS
    for pl in paranoia_levels:
        print('[INFO] Extracting features for PL {}...'.format(pl))
        
        ms_adv_loader = DataLoader(
            malicious_path  = ms_adv_pl_paths[pl-1],
            legitimate_path = legitimate_test_path
        )
        adv_inf_svm_loader = DataLoader(
            malicious_path  = adv_infsvm_pl_paths[pl-1],
            legitimate_path = legitimate_test_path
        )
        adv_log_reg_l1_loader = DataLoader(
            malicious_path  = adv_log_reg_l1_paths[pl-1],
            legitimate_path = legitimate_test_path
        )
        adv_log_reg_l2_loader = DataLoader(
            malicious_path  = adv_log_reg_l2_paths[pl-1],
            legitimate_path = legitimate_test_path
        )
        adv_svm_linear_l1_loader = DataLoader(
            malicious_path  = adv_svm_linear_l1_paths[pl-1],
            legitimate_path = legitimate_test_path
        )
        adv_svm_linear_l2_loader = DataLoader(
            malicious_path  = adv_svm_linear_l2_paths[pl-1],
            legitimate_path = legitimate_test_path
        )
        adv_rf_loader = DataLoader(
            malicious_path  = adv_rf_paths[pl-1],
            legitimate_path = legitimate_test_path
        )

        ms_adv_test_data            = ms_adv_loader.load_data()
        adv_inf_svm_test_data       = adv_inf_svm_loader.load_data()
        adv_log_reg_l1_test_data    = adv_log_reg_l1_loader.load_data()
        adv_log_reg_l2_test_data    = adv_log_reg_l2_loader.load_data()
        adv_svm_linear_l1_test_data = adv_svm_linear_l1_loader.load_data()
        adv_svm_linear_l2_test_data = adv_svm_linear_l2_loader.load_data()
        adv_rf_test_data            = adv_rf_loader.load_data()
        
        extractor = ModSecurityFeaturesExtractor(
            crs_ids_path = crs_ids_path,
            crs_path     = crs_dir,
            crs_pl       = pl
        )

        xts                  , yts                   = extractor.extract_features(test_data)
        ms_adv_xts           , ms_adv_yts            = extractor.extract_features(ms_adv_test_data)
        adv_inf_svm_xts      , adv_inf_svm_yts       = extractor.extract_features(adv_inf_svm_test_data)
        adv_log_reg_l1_xts   , adv_log_reg_l1_yts    = extractor.extract_features(adv_log_reg_l1_test_data)
        adv_log_reg_l2_xts   , adv_log_reg_l2_yts    = extractor.extract_features(adv_log_reg_l2_test_data)
        adv_svm_linear_l1_xts, adv_svm_linear_l1_yts = extractor.extract_features(adv_svm_linear_l1_test_data)
        adv_svm_linear_l2_xts, adv_svm_linear_l2_yts = extractor.extract_features(adv_svm_linear_l2_test_data)
        adv_rf_xts           , adv_rf_yts            = extractor.extract_features(adv_rf_test_data)
        
        for model_name in other_models:
            print('[INFO] Evaluating {} model for PL {}...'.format(model_name, pl))
                
            if model_name == 'modsec':
                label_legend    = 'ModSec'
                color           = 'red'
                ms_adv_settings = {'color': 'red', 'linestyle': 'dashed'}
                
                waf = PyModSecurity(
                    rules_dir = crs_dir,
                    pl        = pl
                )
                y_scores        = waf.predict(test_data['payload'])
                ms_adv_y_scores = waf.predict(ms_adv_test_data['payload'])
                
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
                plot_roc(
                    ms_adv_yts,
                    ms_adv_y_scores,
                    label_legend       = '' ,
                    ax                 = axs.flatten()[pl-1] ,
                    settings           = ms_adv_settings ,
                    plot_rand_guessing = False ,
                    log_scale          = True ,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False ,
                    zoom_axs           = zoom_axs ,
                    pl                 = pl
                )
  
            elif model_name == 'infsvm':
                label_legend = f'SecSVM'
                color        = 'darkmagenta'
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
                    settings           = {'color': color},
                    plot_rand_guessing = False,
                    log_scale          = True,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False,
                    zoom_axs           = zoom_axs,
                    pl                 = pl
                )                 
                plot_roc(
                    adv_inf_svm_yts,
                    adv_y_scores,
                    label_legend       = '' ,
                    ax                 = axs.flatten()[pl-1] ,
                    settings           = adv_settings ,
                    plot_rand_guessing = False ,
                    log_scale          = True ,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False ,
                    zoom_axs           = zoom_axs ,
                    pl                 = pl
                )

            elif model_name == 'rf':
                label_legend = 'RF'
                color        = 'green'
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
                    settings           = {'color': color},
                    plot_rand_guessing = False,
                    log_scale          = True,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False,
                    zoom_axs           = zoom_axs,
                    pl                 = pl
                )
                plot_roc(
                    adv_rf_yts,
                    adv_y_scores,
                    label_legend       = '' ,
                    ax                 = axs.flatten()[pl-1] ,
                    settings           = adv_settings ,
                    plot_rand_guessing = False ,
                    log_scale          = True ,
                    update_roc_values  = True if pl == 1 else False,
                    include_zoom       = False ,
                    zoom_axs           = zoom_axs ,
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
                        os.path.join(models_path, 'linear_svc_pl{}_{}.joblib'.format(pl,penalty))
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
                        zoom_axs           = zoom_axs,
                        pl                 = pl
                    )   
                    plot_roc(
                        adv_svm_linear_l1_yts if penalty == 'l1' else adv_svm_linear_l2_yts,
                        adv_y_scores,
                        label_legend       = '',
                        ax                 = axs.flatten()[pl-1] ,
                        settings           = adv_settings ,
                        plot_rand_guessing = False ,
                        log_scale          = True ,
                        update_roc_values  = True if pl == 1 else False,
                        include_zoom       = False ,
                        zoom_axs           = zoom_axs ,
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
                        zoom_axs           = zoom_axs,
                        pl                 = pl
                    )   
                    plot_roc(
                        adv_log_reg_l1_yts if penalty == 'l1' else adv_log_reg_l2_yts,
                        adv_y_scores,
                        label_legend       = '' ,
                        ax                 = axs.flatten()[pl-1] ,
                        settings           = adv_settings ,
                        plot_rand_guessing = False ,
                        log_scale          = True ,
                        update_roc_values  = True if pl == 1 else False,
                        include_zoom       = False ,
                        zoom_axs           = zoom_axs ,
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
            os.path.join(figures_path, 'roc_curves_test_adv_ds1.pdf'),
            dpi         = 600,
            format      = 'pdf',
            bbox_inches = "tight"
        )