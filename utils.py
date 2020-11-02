import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import yadisk

from tqdm.auto import tqdm


sns.set_theme()


def download_data_if_not_exists():
    """скачать файлы в папку data в рабочей директории"""
    if not os.path.exists('./data'):
        os.mkdir('./data')
        y = yadisk.YaDisk()
        files = [
            'case_presentation.pptx',
            'column_breakdown.txt',
            'models.xlsx',
            'partners.xlsx',
        ]
        for file in files:
            y.download_public(
                'https://yadi.sk/d/...', 
                f'./data/{file}', 
                path=f'/{file}'
            )


def calc_stats_models(df_models):
    """посчитать defect_ratio и cancel_ratio моделей"""
    df_models['defect_ratio'] = df_models['trips_defect_cnt'] \
        / df_models['trips_rated_cnt']
    df_models['cancel_ratio'] = df_models['trips_cancel_cnt'] \
        / (df_models['trips_success_cnt'] + df_models['trips_cancel_cnt'])

    return df_models


def clf_by_defect_ratio(df_models, p_sticking, defect_ratio_threshold=0.1):
    """классифицировать модели по порогу defect_ratio и спрогнозировать число обклеенных"""
    df_models['can_be_branded_new'] = df_models['defect_ratio'] <= defect_ratio_threshold
    
    # согласно предположению I: N_sticked = N_cars * p(sticked)
    df_models['car_sticker_cnt_new'] = 0
    df_models.loc[df_models['can_be_branded_new'], 'car_sticker_cnt_new'] = \
        df_models['car_cnt'][df_models['can_be_branded_new']] * p_sticking
    
    return df_models


def compare_clf_to_baseline(df_models, clf_postfix='_new'):
    """посчитать метрики в сравнении с бейзлайном
    clf_postfix: постфикс для выбора колонки с классификацией / числом брендированных машин
        'can_be_branded' + clf_postfix -- колонка с классификацией, например 'can_be_branded_new'
    """
    clf_can_be_branded_col = 'can_be_branded' + clf_postfix
    clf_car_sticker_cnt_col = 'car_sticker_cnt' + clf_postfix

    old_mask = df_models['can_be_branded']
    new_mask = df_models[clf_can_be_branded_col]
    only_new_mask = ~old_mask & new_mask
    old_and_new_mask = old_mask & new_mask

    # какая доля машин от общего числа станет доступна дополнительно к уже оклеенным
    # при условии, что мы оставим наклейки на всех старых авто
    keep_share = df_models['car_cnt'][old_mask | new_mask].sum() / df_models['car_cnt'].sum()
    
    # какая доля машин от общего числа станет доступна всего при условии, 
    # при условии, что мы снимаем наклейки с части старых авто
    remove_share = df_models['car_cnt'][new_mask].sum() / df_models['car_cnt'].sum()

    # какая доля машин от общего числа станет доступна дополнительно
    additional_share = df_models['car_cnt'][only_new_mask].sum() / df_models['car_cnt'].sum()

    sum_defect_ratio_on_old = (
        df_models['defect_ratio'][old_mask] * 
        df_models['car_sticker_cnt'][old_mask]
    ).sum()
    sum_defect_ratio_on_old_and_new = (
        df_models['defect_ratio'][old_and_new_mask] * 
        df_models['car_sticker_cnt'][old_and_new_mask]
    ).sum()
    expected_sum_defect_ratio_on_new = (
        df_models['defect_ratio'][new_mask] * 
        df_models[clf_car_sticker_cnt_col][new_mask]
    ).sum()
    expected_sum_defect_ratio_on_only_new = (
        df_models['defect_ratio'][only_new_mask] * 
        df_models[clf_car_sticker_cnt_col][only_new_mask]
    ).sum()
    
    # какой ожидаемый средний defect_rate будет на всех брендированных машинах 
    # при условии, что мы оставим наклейки на всех старых авто
    avg_expected_deffect_ratio_keep = (sum_defect_ratio_on_old + expected_sum_defect_ratio_on_only_new) \
        / (df_models['car_sticker_cnt'][old_mask].sum() + df_models[clf_car_sticker_cnt_col][only_new_mask].sum())
    
    # какой ожидаемый средний defect_rate будет на всех брендированных машинах 
    # при условии, что мы снимаем наклейки с части старых авто
    avg_expected_deffect_ratio_remove = (sum_defect_ratio_on_old_and_new + expected_sum_defect_ratio_on_only_new) \
        / (df_models['car_sticker_cnt'][old_and_new_mask].sum() + df_models[clf_car_sticker_cnt_col][only_new_mask].sum())

    # какой ожидаемый средний defect_rate будет только на новых машинах
    avg_expected_deffect_ratio_on_only_new = expected_sum_defect_ratio_on_only_new \
        / df_models[clf_car_sticker_cnt_col][only_new_mask].sum()

    return keep_share, remove_share, additional_share, \
        avg_expected_deffect_ratio_keep, \
        avg_expected_deffect_ratio_remove, \
        avg_expected_deffect_ratio_on_only_new


def gridsearch_defect_ratio_threshold(df_models, p_sticking, threshold_values, print_debug=False):
    """для каждого из порогов почитать метрики в сравнении с бейзлайном"""
    
    stats = {
        'threshold_values': threshold_values, 
        'keep_shares': [], 
        'remove_shares': [], 
        'additional_shares': [],
        'avg_expected_deffect_ratios_keep': [],
        'avg_expected_deffect_ratios_remove': [],
        'avg_expected_deffect_ratios_on_only_new': [],
    }

    for value in tqdm(threshold_values):
        df_models = clf_by_defect_ratio(df_models, p_sticking, defect_ratio_threshold=value)

        keep_share, remove_share, additional_share, \
        avg_expected_deffect_ratio_keep, \
        avg_expected_deffect_ratio_remove, \
        avg_expected_deffect_ratio_on_only_new = \
            compare_clf_to_baseline(df_models, clf_postfix='_new')

        if print_debug:
            print(
                f'threshold == {value}, '
                f'keep_share == {keep_share}, '
                f'remove_share == {remove_share}, '
                f'additional_share == {additional_share}, '
                f'avg_expected_deffect_ratio_keep == {avg_expected_deffect_ratio_keep}, '
                f'avg_expected_deffect_ratio_remove == {avg_expected_deffect_ratio_remove}, '
                f'avg_expected_deffect_ratio_on_only_new == {avg_expected_deffect_ratio_on_only_new}, '
            )
        
        stats['keep_shares'].append(keep_share)
        stats['remove_shares'].append(remove_share)
        stats['additional_shares'].append(additional_share)
        stats['avg_expected_deffect_ratios_keep'].append(avg_expected_deffect_ratio_keep)
        stats['avg_expected_deffect_ratios_remove'].append(avg_expected_deffect_ratio_remove)
        stats['avg_expected_deffect_ratios_on_only_new'].append(avg_expected_deffect_ratio_on_only_new)
    
    return stats


def plot_defect_rate_hist(df_models, clf_postfix=''):
    """построить гистограмы defect_ratio и cancel_ratio для данной классификации
    clf_postfix: постфикс для выбора колонки с классификацией / числом брендированных машин
        'can_be_branded' + clf_postfix -- колонка с классификацией, например 'can_be_branded_new'
    """
    can_be_branded_col = 'can_be_branded' + clf_postfix
    car_sticker_cnt_col = 'car_sticker_cnt' + clf_postfix

    # defect_ratio & cancel_ratio
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
    fig.suptitle('defect_ratio & cancel_ratio hists (density)')

    # can_be_branded & branded | can_be_branded & ~branded | ~can_be_branded
    x_max = df_models['defect_ratio'].quantile(0.99)
    bins = np.linspace(0, x_max, 20)
    
    ax = axes[0]
    ax.set_title('defect_ratio')
    ax.set_xlim((0, x_max))
    ax.hist(
        [
            df_models[df_models[can_be_branded_col]]['defect_ratio'], 
            df_models[df_models[can_be_branded_col]]['defect_ratio'], 
            df_models[~df_models[can_be_branded_col]]['defect_ratio'], 
        ], 
        weights=[
            df_models[df_models[can_be_branded_col]][car_sticker_cnt_col],
            df_models[df_models[can_be_branded_col]]['car_cnt'] - df_models[df_models[can_be_branded_col]][car_sticker_cnt_col],
            df_models[~df_models[can_be_branded_col]]['car_cnt'],
        ],
        density=True,
        bins=bins,
        label=['can_be_branded & branded', 'can_be_branded & ~branded', '~can_be_branded']
    )
    ax.legend()

    x_max = df_models['defect_ratio'].quantile(0.99)
    bins = np.linspace(0, x_max, 20)

    ax = axes[1]
    ax.set_title('cancel_ratio')
    ax.set_xlim((0, x_max))
    ax.hist(
        [
            df_models[df_models[can_be_branded_col]]['cancel_ratio'], 
            df_models[df_models[can_be_branded_col]]['cancel_ratio'], 
            df_models[~df_models[can_be_branded_col]]['cancel_ratio'], 
        ], 
        weights=[
            df_models[df_models[can_be_branded_col]][car_sticker_cnt_col],
            df_models[df_models[can_be_branded_col]]['car_cnt'] - df_models[df_models[can_be_branded_col]][car_sticker_cnt_col],
            df_models[~df_models[can_be_branded_col]]['car_cnt'],
        ],
        density=True,
        bins=bins,
        label=['can_be_branded & branded', 'can_be_branded & ~branded', '~can_be_branded']
    )
    ax.legend()


def plot_shares(stats, current_defect_ratio, current_share, ax=None):
    """построить долю брендированных машин в зависимости от порога"""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.set_ylim((0, 1.2))
    ax.set_title('Доля (прогнозируемая) обклеенных машин')
    ax.set_xlabel('Максимальный допустимый defect_ratio')
    ax.set_ylabel('Доля')
    ax.hlines([current_share], 0, 0.1, colors='r', label='Текущее значение')
    ax.vlines([current_defect_ratio], 0, 1.2, 'k', 'dashed',  alpha=0.6, label='Текущий defect_ratio')
    ax.plot(stats['threshold_values'], stats['keep_shares'], label='Новый | оставляем старые')
    ax.plot(stats['threshold_values'], stats['remove_shares'], label='Новый | убираем часть старых')
    ax.legend()


def plot_defect_ratios(stats, current_defect_ratio, ax=None, zoom=False):
    """построить средний defect_ratio в зависимости от порога"""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if zoom:
        ax.set_ylim((0.020, 0.025))
    else:
        ax.set_ylim((0, 0.025))
    ax.set_title('Defect ratio')
    ax.set_xlabel('Максимальный допустимый defect_ratio')
    ax.set_ylabel('Defect ratio')
    ax.hlines([current_defect_ratio], 0, 0.1, colors='r', label='Текущее значение')
    ax.vlines([current_defect_ratio], 0, 0.025, 'k', 'dashed', alpha=0.6, label='Текущий defect_ratio')
    ax.plot(stats['threshold_values'], stats['avg_expected_deffect_ratios_keep'], label='Новый | оставляем старые')
    ax.plot(stats['threshold_values'], stats['avg_expected_deffect_ratios_remove'], label='Новый | убираем часть старых')
    ax.legend()


def plot_stats(current_share, current_defect_ratio, stats):
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 10), sharex=True)

    xticks = np.arange(0, 0.1 + 0.01, 0.01)
    axes[0].set_xticks(xticks)
    axes[1].set_xticks(xticks)

    plot_defect_ratios(stats, current_defect_ratio, ax=axes[0], zoom=False)
    plot_shares(stats, current_defect_ratio, current_share, ax=axes[1])


if __name__ == '__main__':
    pass
