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
    test_adv         = False
    settings         = toml.load('config.toml')
    crs_dir          = settings['crs_dir']
    crs_ids_path     = settings['crs_ids_path']
    models_path      = settings['models_path']
    figures_path     = settings['figures_path']
    dataset_path     = settings['dataset_path']
    paranoia_levels  = settings['params']['paranoia_levels'] if test_adv else [4]
    models           = settings['params']['models']
    other_models     = settings['params']['other_models']
    penalties        = settings['params']['penalties']
    fig, axs         = plt.subplots(2, 2)
    zoom_axs         = dict()
    
    
    # LOADING DATASET PHASE
    print('[INFO] Loading dataset...')

    legitimate_train_path = os.path.join(dataset_path, 'legitimate_train.json')
    malicious_train_path  = os.path.join(dataset_path, 'malicious_train.json')
    legitimate_test_path  = os.path.join(dataset_path, 'legitimate_test.json')
    malicious_test_path   = os.path.join(dataset_path, 'malicious_test.json')
    adv_infsvm_pl_paths = [
        os.path.join(dataset_path, f'adv_test_inf_svm_pl{pl}_rs20_100rounds.json') for pl in range(1, 5)
    ]
    ms_adv_pl_paths = [
        os.path.join(dataset_path, f'adv_test_ms_pl{pl}_rs20_100rounds.json') for pl in range(1, 5)
    ]
    adv_log_reg_paths = [ 
        os.path.join(dataset_path, f'adv_test_log_reg_pl{pl}_rs20_100rounds.json') for pl in range(1, 5)
    ]
    adv_svm_linear_paths = [ 
        os.path.join(dataset_path, f'adv_test_svm_linear_pl{pl}_rs20_100rounds.json') for pl in range(1, 5)
    ]
    
    adv_rf_paths = [
        os.path.join(dataset_path, f'adv_test_rf_pl{pl}_rs20_100rounds.json') for pl in range(1, 5)
    ]
    
    
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
    for pl in paranoia_levels:
        print('[INFO] Extracting features for PL {}...'.format(pl))
        
        ms_adv_loader = DataLoader(
                    malicious_path=ms_adv_pl_paths[pl-1],
                    legitimate_path=legitimate_test_path
        )
        adv_inf_svm_loader = DataLoader(
                    malicious_path=adv_infsvm_pl_paths[pl-1],
                    legitimate_path=legitimate_test_path
                )
        adv_log_reg_loader = DataLoader(
                    malicious_path=adv_log_reg_paths[pl-1],
                    legitimate_path=legitimate_test_path
                )
        adv_svm_linear_loader = DataLoader(
                    malicious_path=adv_svm_linear_paths[pl-1],
                    legitimate_path=legitimate_test_path
                )
        adv_rf_loader = DataLoader(
                    malicious_path=adv_rf_paths[pl-1],
                    legitimate_path=legitimate_test_path
                )
        
        ms_adv_test_data = ms_adv_loader.load_data()
        adv_inf_svm_test_data = adv_inf_svm_loader.load_data()
        adv_log_reg_test_data = adv_log_reg_loader.load_data()
        adv_svm_linear_test_data = adv_svm_linear_loader.load_data()
        adv_rf_test_data = adv_rf_loader.load_data()
        
        extractor = ModSecurityFeaturesExtractor(
            crs_ids_path = crs_ids_path,
            crs_path     = crs_dir,
            crs_pl       = pl
        )

        xts, yts = extractor.extract_features(test_data)
        ms_adv_xts, ms_adv_yts = extractor.extract_features(ms_adv_test_data)
        adv_inf_svm_xts, adv_inf_svm_yts = extractor.extract_features(adv_inf_svm_test_data)
        adv_log_reg_xts, adv_log_reg_yts = extractor.extract_features(adv_log_reg_test_data)
        adv_svm_linear_xts, adv_svm_linear_yts = extractor.extract_features(adv_svm_linear_test_data)
        adv_rf_xts, adv_rf_yts = extractor.extract_features(adv_rf_test_data)
        
        if test_adv:
            
            for model_name in other_models:
                print('[INFO] Evaluating {} model for PL {}...'.format(model_name, pl))
                    
                if model_name == 'modsec':
                    label_legend = 'ModSec'
                    color        = 'red'
                    ms_adv_label_legend = f'ModSec Model'
                    ms_adv_settings = {'color': 'red', 'linestyle': 'dashed'}
                    waf = PyModSecurity(
                        rules_dir = crs_dir,
                        pl        = pl
                    )
                    y_scores = waf.predict(test_data['payload'])
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
                        label_legend=ms_adv_label_legend,
                        ax=axs.flatten()[pl-1],
                        settings=ms_adv_settings,
                        plot_rand_guessing=False,
                        log_scale=True,
                        update_roc_values=True if pl == 1 else False,
                        include_zoom=False,
                        zoom_axs=zoom_axs,
                        pl=pl
                    )

                    
                elif model_name == 'infsvm':
                    label_legend  = f'InfSVM'
                    color = 'darkmagenta'
                    adv_label_legend = f'Adv InfSVM'
                    adv_settings = {'color': 'darkmagenta', 'linestyle': 'dashed'}
                    model         = joblib.load(
                        os.path.join(models_path, 'inf_svm_pl{}_t1.joblib'.format(pl))
                    )
                    y_scores     = model.decision_function(xts)
                    adv_y_scores = model.decision_function(adv_inf_svm_xts)
                    # print("coef\n",model.coef_)
                    # print("intercept\n",model.intercept_)
                    
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
                        label_legend=adv_label_legend,
                        ax=axs.flatten()[pl-1],
                        settings=adv_settings,
                        plot_rand_guessing=False,
                        log_scale=True,
                        update_roc_values=True if pl == 1 else False,
                        include_zoom=False,
                        zoom_axs=zoom_axs,
                        pl=pl
                    )
                elif model_name == 'rf':
                    label_legend = 'RF'
                    color        = 'green'
                    adv_label_legend = f'Adv RF'
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
                        label_legend=adv_label_legend,
                        ax=axs.flatten()[pl-1],
                        settings=adv_settings,
                        plot_rand_guessing=False,
                        log_scale=True,
                        update_roc_values=True if pl == 1 else False,
                        include_zoom=False,
                        zoom_axs=zoom_axs,
                        pl=pl
                    )
                        
            for model_name in models:
                print('[INFO] Evaluating {} model for PL {}...'.format(model_name, pl))
                        
                if model_name == 'svc':
                    label_legend  = f'SVM'
                    settings      = {'color': 'blue', 'linestyle': 'solid'}
                    adv_label_legend = f'Adv SVM'
                    adv_settings = {'color': 'blue', 'linestyle': 'dashed'}
                    model         = joblib.load(
                        os.path.join(models_path, 'linear_svc_pl{}_l1.joblib'.format(pl))
                    )
                    y_scores     = model.decision_function(xts)
                    adv_y_scores = model.decision_function(adv_svm_linear_xts)
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
                        adv_svm_linear_yts,
                        adv_y_scores,
                        label_legend=adv_label_legend,
                        ax=axs.flatten()[pl-1],
                        settings=adv_settings,
                        plot_rand_guessing=False,
                        log_scale=True,
                        update_roc_values=True if pl == 1 else False,
                        include_zoom=False,
                        zoom_axs=zoom_axs,
                        pl=pl
                    )
                elif model_name == 'log_reg':
                    label_legend  = f'LR'
                    settings      = {'color': 'orange', 'linestyle': 'solid'}
                    adv_label_legend = f'Adv LR'
                    adv_settings = {'color': 'orange', 'linestyle': 'dashed'}
                    model         = joblib.load(
                        os.path.join(models_path, 'log_reg_pl{}_l1.joblib'.format(pl))
                    )
                    y_scores      = model.predict_proba(xts)[:, 1]
                    adv_y_scores  = model.predict_proba(adv_log_reg_xts)[:, 1]
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
                        adv_log_reg_yts,
                        adv_y_scores,
                        label_legend=adv_label_legend,
                        ax=axs.flatten()[pl-1],
                        settings=adv_settings,
                        plot_rand_guessing=False,
                        log_scale=True,
                        update_roc_values=True if pl == 1 else False,
                        include_zoom=False,
                        zoom_axs=zoom_axs,
                        pl=pl
                    )
            # Final global settings for the figure
            for idx, ax in enumerate(axs.flatten()):
                ax.set_title('PL {}'.format(idx+1), fontsize=20)
                ax.xaxis.set_tick_params(labelsize = 15)
                ax.yaxis.set_tick_params(labelsize = 15)
                ax.xaxis.label.set_size(20)
                ax.yaxis.label.set_size(20)

            handles, labels = axs.flatten()[0].get_legend_handles_labels()      
            
            fig.legend(
                handles, 
                labels,
                loc            = 'upper center',
                bbox_to_anchor = (0.5, -0.01),
                fancybox       = True,
                shadow         = True,
                ncol           = 6,
                fontsize       = 20
            )
            fig.set_size_inches(16, 10)
            fig.tight_layout(pad = 2.0)
            fig.savefig(
                os.path.join(figures_path, 'roc_curves_test_adv.pdf'),
                dpi         = 600,
                format      = 'pdf',
                bbox_inches = "tight"
            )

        else:  
            adv_infsvm_pl_path = os.path.join(dataset_path, 'adv_train_test_inf_svm_pl4_rs20_100rounds.json')
            adv_log_reg_path = os.path.join(dataset_path, 'adv_train_test_log_reg_pl4_rs20_100rounds.json')
            adv_svm_linear_path = os.path.join(dataset_path, 'adv_train_test_svm_linear_pl4_rs20_100rounds.json')
            adv_rf_path = os.path.join(dataset_path, 'adv_train_test_rf_pl4_rs20_100rounds.json')
            
            adv_inf_svm_loader = DataLoader(
                        malicious_path=adv_infsvm_pl_path,
                        legitimate_path=legitimate_test_path
                    )
            adv_log_reg_loader = DataLoader(
                        malicious_path=adv_log_reg_path,
                        legitimate_path=legitimate_test_path
                    )
            adv_svm_linear_loader = DataLoader(
                        malicious_path=adv_svm_linear_path,
                        legitimate_path=legitimate_test_path
                    )
            adv_rf_loader = DataLoader(
                        malicious_path=adv_rf_path,
                        legitimate_path=legitimate_test_path
                    )
            
            ms_adv_test_data = ms_adv_loader.load_data()
            adv_inf_svm_test_data = adv_inf_svm_loader.load_data()
            adv_log_reg_test_data = adv_log_reg_loader.load_data()
            adv_svm_linear_test_data = adv_svm_linear_loader.load_data()
            adv_rf_test_data = adv_rf_loader.load_data()
            
            extractor = ModSecurityFeaturesExtractor(
                crs_ids_path = crs_ids_path,
                crs_path     = crs_dir,
                crs_pl       = pl
            )

            xts, yts = extractor.extract_features(test_data)
            adv_inf_svm_xts, adv_inf_svm_yts = extractor.extract_features(adv_inf_svm_test_data)
            adv_log_reg_xts, adv_log_reg_yts = extractor.extract_features(adv_log_reg_test_data)
            adv_svm_linear_xts, adv_svm_linear_yts = extractor.extract_features(adv_svm_linear_test_data)
            adv_rf_xts, adv_rf_yts = extractor.extract_features(adv_rf_test_data)
            
            model_settings = {
                'svc': {
                    'label': 'Linear SVM',
                    'color': 'orange',
                    'adv_color': 'deepskyblue',
                    'model': joblib.load(os.path.join(models_path, 'linear_svc_pl4_l1.joblib')),
                    'adv_model': joblib.load(os.path.join(models_path, 'adv_linear_svc_pl4_l1.joblib')),
                    'adv_xts': adv_svm_linear_xts,
                    'adv_yts': adv_svm_linear_yts
                },
                'rf': {
                    'label': 'Random Forest',
                    'color': 'orange',
                    'adv_color': 'deepskyblue',
                    'model': joblib.load(os.path.join(models_path, 'rf_pl4.joblib')),
                    'adv_model': joblib.load(os.path.join(models_path, 'adv_rf_pl4.joblib')),
                    'adv_xts': adv_rf_xts,
                    'adv_yts': adv_rf_yts
                },
                'log_reg': {
                    'label': 'Logistic Regression',
                    'color': 'orange',
                    'adv_color': 'deepskyblue',
                    'model': joblib.load(os.path.join(models_path, 'log_reg_pl4_l1.joblib')),
                    'adv_model': joblib.load(os.path.join(models_path, 'adv_log_reg_pl4_l1.joblib')),
                    'adv_xts': adv_log_reg_xts,
                    'adv_yts': adv_log_reg_yts
                },
                'infsvm': {
                    'label': 'Inf SVM',
                    'color': 'orange',
                    'adv_color': 'deepskyblue',
                    'model': joblib.load(os.path.join(models_path, 'inf_svm_pl4_t0.5.joblib')),
                    'adv_xts': adv_inf_svm_xts,
                    'adv_yts': adv_inf_svm_yts
                }
            }     
            
            
            for idx, (model_name, settings) in enumerate(model_settings.items()):
                ax = axs[idx // 2, idx % 2] 

                print(f'[INFO] Evaluating {model_name} model for PL 4...')
                
                y_scores = settings['model'].decision_function(xts) if model_name in ['svc', 'infsvm'] else settings['model'].predict_proba(xts)[:, 1]
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
                
                adv_y_scores = settings['model'].decision_function(settings['adv_xts']) if model_name in ['svc', 'infsvm'] else settings['model'].predict_proba(settings['adv_xts'])[:, 1]
                plot_roc(
                    settings['adv_yts'],
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
                    adv_trained_y_scores = settings['adv_model'].decision_function(xts) if model_name in ['svc'] else settings['adv_model'].predict_proba(xts)[:, 1]
                    
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
                
                    adv_trained_adv_y_scores = settings['adv_model'].decision_function(settings['adv_xts']) if model_name in ['svc'] else settings['adv_model'].predict_proba(settings['adv_xts'])[:, 1]
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

                ax.set_title(f'{settings["label"]} PL4', fontsize=20)
                ax.xaxis.set_tick_params(labelsize = 15)
                ax.yaxis.set_tick_params(labelsize = 15)
                ax.xaxis.label.set_size(20)
                ax.yaxis.label.set_size(20)

            handles, labels = axs[0,0].get_legend_handles_labels()      

            fig.legend(
                handles, 
                labels,
                loc            = 'upper center',
                bbox_to_anchor = (0.5, -0.01),
                fancybox       = True,
                shadow         = True,
                ncol           = 6,
                fontsize       = 20
            )
            fig.set_size_inches(16, 10)
            fig.tight_layout(pad = 2.0)
            fig.savefig(
                os.path.join(figures_path, 'LLLL.pdf'),
                dpi         = 600,
                format      = 'pdf',
                bbox_inches = "tight"
            )


        
        