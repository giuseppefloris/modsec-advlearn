"""
This script is used to plot the ROC curves for the pretrained ML models and the ModSecurity WAF.
"""

import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    figures_path     = settings['figures_path']
    dataset_path     = settings['dataset_path']
    pl               = 4
    models           = settings['params']['models']
    other_models     = settings['params']['other_models']
    penalties        = settings['params']['penalties']
    fig, axs         = plt.subplots(1, 6)
    zoom_axs         = dict()
    
    
    # LOADING DATASET PHASE
    print('[INFO] Loading dataset...')

    legitimate_train_path = os.path.join(dataset_path, 'legitimate_train.json')
    malicious_train_path  = os.path.join(dataset_path, 'malicious_train.json')
    legitimate_test_path  = os.path.join(dataset_path, 'legitimate_test.json')
    malicious_test_path   = os.path.join(dataset_path, 'malicious_test.json')
    adv_infsvm_paths_test = os.path.join(dataset_path, f'adv_test_inf_svm_pl{pl}_rs20_100rounds.json') 
    adv_log_reg_l1_paths_test = os.path.join(dataset_path, f'adv_test_log_reg_pl{pl}_rs20_100rounds.json') 
    adv_log_reg_l2_paths_test = os.path.join(dataset_path, f'adv_test_log_reg_l2_pl{pl}_rs20_100rounds.json') 
    adv_svm_linear_l1_paths_test = os.path.join(dataset_path, f'adv_test_svm_linear_pl{pl}_rs20_100rounds.json') 
    adv_svm_linear_l2_paths_test = os.path.join(dataset_path, f'adv_test_svm_linear_l2_pl{pl}_rs20_100rounds.json') 
    adv_rf_paths_test = os.path.join(dataset_path, f'adv_test_rf_pl{pl}_rs20_100rounds.json') 
    
    
    loader                = DataLoader(
        malicious_path  = malicious_train_path,
        legitimate_path = legitimate_train_path
    )    
    training_data = loader.load_data()

    loader                = DataLoader(
        malicious_path  = malicious_test_path,
        legitimate_path = legitimate_test_path
    )    
    test_data = loader.load_data()
    
    
    # STARTING EXPERIMENTS
   
    adv_inf_svm_loader_test = DataLoader(
                malicious_path=adv_infsvm_paths_test,
                legitimate_path=legitimate_test_path
            )
    adv_log_reg_l1_loader_test = DataLoader(
                malicious_path=adv_log_reg_l1_paths_test,
                legitimate_path=legitimate_test_path
            )
    adv_log_reg_l2_loader_test = DataLoader(
                malicious_path=adv_log_reg_l2_paths_test,
                legitimate_path=legitimate_test_path
            )
    adv_svm_linear_l1_loader_test = DataLoader(
                malicious_path=adv_svm_linear_l1_paths_test,
                legitimate_path=legitimate_test_path
            )
    adv_svm_linear_l2_loader_test = DataLoader(
                malicious_path=adv_svm_linear_l2_paths_test,
                legitimate_path=legitimate_test_path
            )
    adv_rf_loader_test = DataLoader(
                malicious_path=adv_rf_paths_test,
                legitimate_path=legitimate_test_path
            )
    
    
    adv_inf_svm_test_data_test = adv_inf_svm_loader_test.load_data()
    adv_log_reg_l1_test_data_test = adv_log_reg_l1_loader_test.load_data()
    adv_log_reg_l2_test_data_test = adv_log_reg_l2_loader_test.load_data()
    adv_svm_linear_l1_test_data_test = adv_svm_linear_l1_loader_test.load_data()
    adv_svm_linear_l2_test_data_test = adv_svm_linear_l2_loader_test.load_data()
    adv_rf_test_data_test = adv_rf_loader_test.load_data()
    
    extractor = ModSecurityFeaturesExtractor(
        crs_ids_path = crs_ids_path,
        crs_path     = crs_dir,
        crs_pl       = pl
    )

    xts, yts = extractor.extract_features(test_data)
    inf_svm_xts, inf_svm_yts = extractor.extract_features(adv_inf_svm_test_data_test)
    log_reg_l1_xts, log_reg_l1_yts = extractor.extract_features(adv_log_reg_l1_test_data_test)
    log_reg_l2_xts, log_reg_l2_yts = extractor.extract_features(adv_log_reg_l2_test_data_test)
    svm_linear_l1_xts, svm_linear_l1_yts = extractor.extract_features(adv_svm_linear_l1_test_data_test)
    svm_linear_l2_xts, svm_linear_l2_yts = extractor.extract_features(adv_svm_linear_l2_test_data_test)
    rf_xts, rf_yts = extractor.extract_features(adv_rf_test_data_test)
    
    adv_infsvm_pl_path = os.path.join(dataset_path, 'adv_train_test_inf_svm_pl4_rs20_100rounds.json')
    adv_log_reg_l1_path = os.path.join(dataset_path, 'adv_train_test_log_reg_pl4_rs20_100rounds.json')
    adv_log_reg_l2_path = os.path.join(dataset_path, 'adv_train_test_log_reg_l2_pl4_rs20_100rounds.json')
    adv_svm_linear_l1_path = os.path.join(dataset_path, 'adv_train_test_svm_linear_pl4_rs20_100rounds.json')
    adv_svm_linear_l2_path = os.path.join(dataset_path, 'adv_train_test_svm_linear_l2_pl4_rs20_100rounds.json')
    adv_rf_path = os.path.join(dataset_path, 'adv_train_test_rf_pl4_rs20_100rounds.json')
    
    adv_inf_svm_loader = DataLoader(
                malicious_path=adv_infsvm_pl_path,
                legitimate_path=legitimate_test_path
            )
    adv_log_reg_l1_loader = DataLoader(
                malicious_path=adv_log_reg_l1_path,
                legitimate_path=legitimate_test_path
            )
    adv_log_reg_l2_loader = DataLoader(
                malicious_path=adv_log_reg_l2_path,
                legitimate_path=legitimate_test_path
            )
    adv_svm_linear_l1_loader = DataLoader(
                malicious_path=adv_svm_linear_l1_path,
                legitimate_path=legitimate_test_path
            )
    adv_svm_linear_l2_loader = DataLoader(
                malicious_path=adv_svm_linear_l2_path,
                legitimate_path=legitimate_test_path
            )
    adv_rf_loader = DataLoader(
                malicious_path=adv_rf_path,
                legitimate_path=legitimate_test_path
            )
    
    
    adv_inf_svm_test_data = adv_inf_svm_loader.load_data()
    adv_log_reg_l1_test_data = adv_log_reg_l1_loader.load_data()
    adv_log_reg_l2_test_data = adv_log_reg_l2_loader.load_data()
    adv_svm_linear_l1_test_data = adv_svm_linear_l1_loader.load_data()
    adv_svm_linear_l2_test_data = adv_svm_linear_l2_loader.load_data()
    adv_rf_test_data = adv_rf_loader.load_data()
    
    extractor = ModSecurityFeaturesExtractor(
        crs_ids_path = crs_ids_path,
        crs_path     = crs_dir,
        crs_pl       = pl
    )

    xts, yts = extractor.extract_features(test_data)
    adv_inf_svm_xts, adv_inf_svm_yts = extractor.extract_features(adv_inf_svm_test_data)
    adv_log_reg_l1_xts, adv_log_reg_l1_yts = extractor.extract_features(adv_log_reg_l1_test_data)
    adv_log_reg_l2_xts, adv_log_reg_l2_yts = extractor.extract_features(adv_log_reg_l2_test_data)
    adv_svm_linear_l1_xts, adv_svm_linear_l1_yts = extractor.extract_features(adv_svm_linear_l1_test_data)
    adv_svm_linear_l2_xts, adv_svm_linear_l2_yts = extractor.extract_features(adv_svm_linear_l2_test_data)
    adv_rf_xts, adv_rf_yts = extractor.extract_features(adv_rf_test_data)
    
    model_settings = {
        'svc_l1': {
            'label': 'Linear SVM $\ell_1$',
            'color': 'orange',
            'adv_color': 'deepskyblue',
            'model': joblib.load(os.path.join(models_path, 'linear_svc_pl4_l1.joblib')),
            'adv_model': joblib.load(os.path.join(models_path, 'adv_linear_svc_pl4_l1.joblib')),
            'adv_xts_test': svm_linear_l1_xts,
            'adv_yts_test': svm_linear_l1_yts,
            'adv_xts': adv_svm_linear_l1_xts,
            'adv_yts': adv_svm_linear_l1_yts,
        },
        'svc_l2': {
            'label': 'Linear SVM $\ell_2$',
            'color': 'orange',
            'adv_color': 'deepskyblue',
            'model': joblib.load(os.path.join(models_path, 'linear_svc_pl4_l2.joblib')),
            'adv_model': joblib.load(os.path.join(models_path, 'adv_linear_svc_pl4_l2.joblib')),
            'adv_xts_test': svm_linear_l2_xts,
            'adv_yts_test': svm_linear_l2_yts,
            'adv_xts': adv_svm_linear_l2_xts,
            'adv_yts': adv_svm_linear_l2_yts
        },
        'rf': {
            'label': 'RF',
            'color': 'orange',
            'adv_color': 'deepskyblue',
            'model': joblib.load(os.path.join(models_path, 'rf_pl4.joblib')),
            'adv_model': joblib.load(os.path.join(models_path, 'adv_rf_pl4.joblib')),
            'adv_xts_test': rf_xts,
            'adv_yts_test': rf_yts,
            'adv_xts': adv_rf_xts,
            'adv_yts': adv_rf_yts
        },
        'log_reg_l1': {
            'label': 'LR $\ell_1$',
            'color': 'orange',
            'adv_color': 'deepskyblue',
            'model': joblib.load(os.path.join(models_path, 'log_reg_pl4_l1.joblib')),
            'adv_model': joblib.load(os.path.join(models_path, 'adv_log_reg_pl4_l1.joblib')),
            'adv_xts_test': log_reg_l1_xts,
            'adv_yts_test': log_reg_l1_yts,
            'adv_xts': adv_log_reg_l1_xts,
            'adv_yts': adv_log_reg_l1_yts
        },
        'log_reg_l2': {
            'label': 'LR $\\ell_2$',
            'color': 'orange',
            'adv_color': 'deepskyblue',
            'model': joblib.load(os.path.join(models_path, 'log_reg_pl4_l2.joblib')),
            'adv_model': joblib.load(os.path.join(models_path, 'adv_log_reg_pl4_l2.joblib')),
            'adv_xts_test': log_reg_l2_xts,
            'adv_yts_test': log_reg_l2_yts,
            'adv_xts': adv_log_reg_l2_xts,
            'adv_yts': adv_log_reg_l2_yts
        },
        'infsvm': {
            'label': 'Sec SVM',
            'color': 'orange',
            'adv_color': 'deepskyblue',
            'model': joblib.load(os.path.join(models_path, 'inf_svm_pl4_t0.5.joblib')),
            'adv_xts_test': inf_svm_xts,
            'adv_yts_test': inf_svm_yts,
            'adv_xts': adv_inf_svm_xts,
            'adv_yts': adv_inf_svm_yts
        }
    }     
    
    
    for idx, (model_name, settings) in enumerate(model_settings.items()):
        #ax = axs[idx // 2, idx % 2] 
        ax = axs[idx]
        print(f'[INFO] Evaluating {model_name} model for PL 4...')
        
        y_scores = settings['model'].decision_function(xts) if model_name in ['svc_l1', 'svc_l2', 'infsvm'] else settings['model'].predict_proba(xts)[:, 1]
        plot_roc(
            yts,
            y_scores,
            label_legend='MLModSec test',
            ax=ax,
            settings={'color': settings['color'], 'linestyle': 'solid'},
            plot_rand_guessing=False,
            log_scale=True,
            update_roc_values=False,
            include_zoom=False,
            zoom_axs=zoom_axs,
            pl=pl
        )
        
        adv_y_scores = settings['model'].decision_function(settings['adv_xts_test']) if model_name in ['svc_l1', 'svc_l2', 'infsvm'] else settings['model'].predict_proba(settings['adv_xts_test'])[:, 1]
        plot_roc(
            settings['adv_yts_test'],
            adv_y_scores,
            label_legend='MLModSec test-adv',
            ax=ax,
            settings={'color': settings['color'], 'linestyle': 'dashed'},
            plot_rand_guessing=False,
            log_scale=True,
            update_roc_values=False,
            include_zoom=False,
            zoom_axs=zoom_axs,
            pl=pl
        )
        if model_name != 'infsvm':
            adv_trained_y_scores = settings['adv_model'].decision_function(xts) if model_name in ['svc_l1', 'svc_l2'] else settings['adv_model'].predict_proba(xts)[:, 1]
            
            plot_roc(
                yts,
                adv_trained_y_scores,
                label_legend='AdvModSec test',
                ax=ax,
                settings={'color': settings['adv_color'], 'linestyle': 'solid'},
                plot_rand_guessing=False,
                log_scale=True,
                update_roc_values=False,
                include_zoom=False,
                zoom_axs=zoom_axs,
                pl=pl
            )
        
            adv_trained_adv_y_scores = settings['adv_model'].decision_function(settings['adv_xts']) if model_name in ['svc_l1', 'svc_l2'] else settings['adv_model'].predict_proba(settings['adv_xts'])[:, 1]
            plot_roc(
                settings['adv_yts'],
                adv_trained_adv_y_scores,
                label_legend='AdvModSec test-adv',
                ax=ax,
                settings={'color': settings['adv_color'], 'linestyle': 'dashed'},
                plot_rand_guessing=False,
                log_scale=True,
                update_roc_values=False,
                include_zoom=False,
                zoom_axs=zoom_axs,
                pl=pl
            )

    
# Final global settings for the figure

        ax.set_title(f'{settings["label"]} PL4', fontsize=16)
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
        os.path.join(figures_path, 'DS1.pdf'),
        dpi         = 600,
        format      = 'pdf',
        bbox_inches = "tight"
    )