"""
This script is used to plot the activation probability of the OWASP CRS rules.
"""

import matplotlib.pyplot as plt
import os
import json
import numpy as np
import pandas as pd
import seaborn.objects as so
import toml
import sys
import pickle # only for waf-a-mole dataset
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import DataLoader
from src.extractor import ModSecurityFeaturesExtractor




def analyze_rules_importance(
    owasp_crs_rules_ids_filepath: str,
    figs_save_path              : str,
    benign_load_path            : str,
    attacks_load_path           : str,
    adv_payloads_filename       : str,
    pl               = 4,
    rules_selector   = None,
    legend_fontsize  = 13,
    axis_labels_size = 16,
    tick_labels_size = 14
):
    with open(owasp_crs_rules_ids_filepath, 'r') as fp:
        data = json.load(fp)
        owasp_crs_ids = sorted(data['rules_ids'])

    info_rules               = [942011, 942012, 942013, 942014, 942015, 942016, 942017, 942018]
    crs_rules_ids_pl1_unique = [942100, 942140, 942151, 942160, 942170, 942190, 942220, 942230, 942240, 942250, 942270, 942280, 942290, 942320, 942350, 942360, 942500, 942540, 942560, 942550]
    crs_rules_ids_pl3_unique = [942251, 942490, 942420, 942431, 942460, 942511, 942530]
    crs_rules_ids_pl4_unique = [942421, 942432]
    crs_rules_ids_pl2_unique = list(set([int(rule) for rule in owasp_crs_ids]) - set(crs_rules_ids_pl1_unique + crs_rules_ids_pl3_unique + crs_rules_ids_pl4_unique + info_rules))

    crs_rules_ids_pl1 = crs_rules_ids_pl1_unique
    crs_rules_ids_pl2 = crs_rules_ids_pl1 + crs_rules_ids_pl2_unique
    crs_rules_ids_pl3 = crs_rules_ids_pl2 + crs_rules_ids_pl3_unique
    crs_rules_ids_pl4 = crs_rules_ids_pl3 + crs_rules_ids_pl4_unique

    num_samples_adv  = 2001
    num_samples_base = 5000

    assert 0 < pl < 5

    if pl == 1:
        rules_filter = crs_rules_ids_pl1
    elif pl == 2:
        rules_filter = crs_rules_ids_pl2
    elif pl == 3:
        rules_filter = crs_rules_ids_pl3
    elif pl == 4:
        rules_filter = crs_rules_ids_pl4

    loader = DataLoader(
        malicious_path  = attacks_load_path,
        legitimate_path = benign_load_path
    )
    adv_loader = DataLoader(
        malicious_path  = adv_payloads_filename,
        legitimate_path = benign_load_path
    )
    data_adv = adv_loader.load_data()
    dataset  = loader.load_data()
    
    extractor = ModSecurityFeaturesExtractor(
        crs_ids_path = crs_ids_path,
        crs_path     = crs_dir,
        crs_pl       = pl
    )

    xts, yts    = extractor.extract_features(dataset)
    adv_dataset = data_adv[data_adv['label'] == 1]
    xts_adv, _  = extractor.extract_features(adv_dataset)
    
    df_benign = pd.DataFrame(data=xts[yts == 0], index=list(range(num_samples_base)), columns=owasp_crs_ids)
    df_attack = pd.DataFrame(data=xts[yts == 1], index=list(range(num_samples_base)), columns=owasp_crs_ids)
    df_adv    = pd.DataFrame(data=xts_adv, index=list(range(num_samples_adv)), columns=owasp_crs_ids)

    # rules to be removed from the plots: rules that are not triggered by any benign, attack and adv. samples
    rules_to_remove = []
    for rule in owasp_crs_ids:
        if df_attack[rule].sum() == 0 and df_benign[rule].sum() == 0 and df_adv[rule].sum(): 
            rules_to_remove.append(rule)
            print(f'SUM {df_adv[rule].sum()} of rule {rule} is zero')
            print("RULES NEVER TRIGGERED: {}".format(rules_to_remove))

    # select rules related to target PL, sort them alphabetically
    select_rules = sorted([rule for rule in owasp_crs_ids if (int(rule) in rules_filter) and (rule not in rules_to_remove)])

    delta = (df_adv[select_rules] - df_attack[select_rules]).mean().values
    assert delta.flatten().shape[0] == len(select_rules)

    rules_delta  = {r: s for r, s in zip(select_rules, delta.tolist())}
    rules_delta  = dict(sorted(rules_delta.items(), key=lambda item: item[1], reverse=False))  # reverse = True
    sorted_rules = list(rules_delta.keys())

    pos_rules, neg_rules, same_rules = list(), list(), list()
    for rule, slope in rules_delta.items():
        if slope > 0:
            pos_rules.append(rule)
        elif slope < 0:
            neg_rules.append(rule)
        else:
            same_rules.append(rule)
    
    if rules_selector is not None and isinstance(rules_selector, list):
        weights                   = np.array([w for w, rule in zip(weights.tolist(), sorted_rules) if rule if rule[3:] in rules_selector])
        weights_advtrain          = np.array([w for w, rule in zip(weights_advtrain.tolist(), sorted_rules) if rule if rule[3:] in rules_selector])
        sorted_rules              = [rule for rule in sorted_rules if rule[3:] in rules_selector]
        rules_activation_filename = 'comparison_attack_adv_svm_new.pdf'
    else:
        rules_activation_filename = 'comparison_attack_adv_svm_l2_ds_wafamole.pdf' # To change

    adv_prob    = df_adv[sorted_rules].mean().values.tolist()
    attack_prob = df_attack[sorted_rules].mean().values.tolist() 
    
    df_plot = pd.DataFrame(
        {
            'rules': sorted_rules * 2,
            'prob' : adv_prob + attack_prob,
            'type' : (['adversarial'] * len(sorted_rules)) + (['malicious'] * len(sorted_rules))
        })

    fig_prob, ax_prob = plt.subplots(1, 1)
    
    so.Plot(df_plot, x='rules', y='prob', color='type')\
        .add(so.Bar(), legend=True)\
        .scale(color=['orange', 'deepskyblue'])\
        .on(ax_prob)\
        .plot()
    
    ax_prob.set_xticklabels(
        [rule[3:] for rule in sorted_rules],
        rotation      = 75,
        ha            = 'right',
        rotation_mode = 'anchor'
    )
    
    legend = fig_prob.legends.pop(0)
    
    ax_prob.legend(
        legend.legendHandles, 
        [t.get_text() for t in legend.texts], 
        loc      = 'upper left',
        fancybox = True,
        shadow   = False,
        fontsize = legend_fontsize
    )
    ax_prob.set_xlabel('CRS SQLi Rules', fontsize=axis_labels_size, labelpad=10)
    ax_prob.set_ylabel('Activation probability', fontsize=axis_labels_size, labelpad=10)
    ax_prob.set_xmargin(0.05)
    ax_prob.set_ymargin(0.05)
    ax_prob.xaxis.set_tick_params(labelsize=tick_labels_size)
    ax_prob.yaxis.set_tick_params(labelsize=tick_labels_size)
    ax_prob.grid(visible=True, axis='both', color='gray', linestyle='dotted')
    
    fig_prob.set_size_inches(16, 6)
    fig_prob.tight_layout()
    fig_prob.savefig(
        os.path.join(figs_save_path, rules_activation_filename), 
        dpi         = 600,
        format      = 'pdf',
        bbox_inches = "tight"
    )


if __name__ == '__main__':
    settings         = toml.load('config.toml')
    crs_ids_path     = settings['crs_ids_path']
    figures_path     = settings['figures_path']
    dataset_path     = settings['dataset_path']
    pl               = settings['params']['paranoia_levels']
    adv_dataset_path = settings['adv_dataset_path']
    crs_dir          = settings['crs_dir']

    adv_examples_base_path       = adv_dataset_path
    benign_load_path             = os.path.join(dataset_path, 'legitimate_test.json')
    attacks_load_path            = os.path.join(dataset_path, 'malicious_test.json')
    adv_payloads_filename        = os.path.join(
        adv_examples_base_path, 
        'adv_test_svm_linear_l2_pl{pl}_rs20_100rounds.json'.format(pl=pl)
    )

    analyze_rules_importance(
        owasp_crs_rules_ids_filepath = crs_ids_path,
        figs_save_path               = figures_path,
        benign_load_path             = benign_load_path,
        attacks_load_path            = attacks_load_path,
        adv_payloads_filename        = adv_payloads_filename,
        legend_fontsize              = 13,
        axis_labels_size             = 18,
        tick_labels_size             = 14
    )