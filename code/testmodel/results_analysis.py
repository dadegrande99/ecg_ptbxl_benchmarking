import numpy as np
from utils.metrics import (
    compute_metrics, compute_diff_and_confidence, metrics_per_confidence,
    custom_entropy_formula_analysis as custom_entropy_formula,
    compute_diff_and_variational
)
from utils import list_directories, create_tensors_from_dataframe, find_files_with_extension
from utils.plotting import (
    plot_exit_counts, boxplot_metrics, barplot_metrics, plot_cumulative_count,
    plot_model_metrics, plot_model_metrics2, plot_dictionary_subplots,
    plot_metric_vs_confidence, plot_metric_on_confidence, plot_metric_trend
)
import json
import os

# Warning settings
import warnings
warnings.filterwarnings("ignore")


def main(output_dir: str = '../../output/', save_plots: bool = True) -> None:
    """
    Main function to analyze the results of the models.
    - Load the configuration of the experiments.
    - Load the models and the metrics.
    - Compute the metrics for the models.
    - Plot the results.

    Parameters
    - output_dir (str): Directory where to save the results. Default is '../../output/'
    - save_plots (bool): if True, save the plots, otherwise, show the plots. Default is True.

    Returns
    - None
    """
    # check if the output directory exists
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Directory {output_dir} does not exist.")
    # configuration for the experiments
    with open(f'{output_dir}config.json', 'r') as f:
        config = json.load(f)
    num_epochs = config['num_epochs']
    # print configuration
    print("Configuration for the experiments:")
    max_key_len = max([len(k) for k in config.keys()])
    for key, value in config.items():
        print(f'- {key}:{(max_key_len - len(key) + 1) * " "} {value}')

    y_datas = {
        'train': create_tensors_from_dataframe(f'{output_dir}/data/y_training.csv'),
        'val': create_tensors_from_dataframe(f'{output_dir}/data/y_validation.csv'),
        'test': create_tensors_from_dataframe(f'{output_dir}/data/y_test.csv'),
        'mcd_validation': create_tensors_from_dataframe(f'{output_dir}/data/y_test.csv')
    }
    y_lens = {mode: len(y_datas[mode][1]) for mode in y_datas}

    # print models
    print("\n\nModels:")
    for el in list_directories(f'{output_dir}/models'):
        print(f'- {el}')

    print("\n\nLoading models ...")

    # train
    models = {el: {'train': {}, 'val': {}, 'test': {}, 'mcd_validation': {}}
              for el in list_directories(f'{output_dir}/models')}
    models_metrics = {el: {'train': {}, 'val': {}, 'test': {}}
                      for el in list_directories(f'{output_dir}/models')}

    # manage exist
    for model_name in models:
        exits = list_directories(f'{output_dir}/models/{model_name}/{"train"}')
        for mode in models[model_name]:
            models[model_name][mode] = {el: {} for el in range(len(exits))}
            models_metrics[model_name][mode] = {
                el: {} for el in range(len(exits))}
    count_exit = {}

    # define metrics
    metrics = ['auc', 'accuracy', 'f1', 'entropy', 'recall']
    for model_name in models_metrics:
        for mode in models_metrics[model_name]:
            for exit in models_metrics[model_name][mode]:
                models_metrics[model_name][mode][exit] = {
                    metric: [] for metric in metrics}

    # read best threshold
    for model_name in models:
        # read values.json
        with open(os.path.join(output_dir, 'models', model_name, 'values.json'), 'r') as f:
            value = json.load(f)
        models[model_name]["best_threshold"] = value["best_threshold"]
        count_exit[model_name] = value["exits"]

    # train & validation
    modes = ['train', 'val']
    for mode in modes:
        for model_name in models:
            for exit in models[model_name][mode]:
                for el in find_files_with_extension(os.path.join(output_dir, 'models', model_name, mode, f'exit_{exit}'), 'csv'):
                    epoch = int(el.split("_")[-1].split(".")[0])
                    models[model_name][mode][exit][epoch] = create_tensors_from_dataframe(
                        os.path.join(output_dir, 'models', model_name, mode, f'exit_{exit}', el))

        # metrics
        for model_name in models_metrics:
            for exit in models_metrics[model_name][mode]:
                for i in models[model_name][mode][exit]:
                    outputs, targets = models[model_name][mode][exit][i]
                    avg_auc, avg_f1, avg_acc, avg_recall = compute_metrics(
                        # type: ignore
                        outputs, targets, models[model_name]["best_threshold"])[:4]
                    models_metrics[model_name][mode][exit]['auc'].append(
                        avg_auc)
                    models_metrics[model_name][mode][exit]['accuracy'].append(
                        avg_acc)
                    models_metrics[model_name][mode][exit]['f1'].append(avg_f1)
                    models_metrics[model_name][mode][exit]['recall'].append(
                        avg_recall)
                    models_metrics[model_name][mode][exit]['entropy'].append(
                        float(custom_entropy_formula(outputs.detach().cpu().numpy())))

    # test & mcd validation
    modes = ['test', 'mcd_validation']
    for mode in modes:
        for model_name in models:
            for exit in models[model_name][mode]:
                models[model_name][mode][exit] = create_tensors_from_dataframe(
                    os.path.join(output_dir, 'models', model_name, mode, f'exit_{exit}.csv'))

    # metrics
    for mode in modes:
        for model_name in models_metrics:
            for exit in models_metrics[model_name][mode]:
                outputs, targets = models[model_name][mode][exit]
                if mode == 'mcd_validation':
                    outputs = outputs.mean(2)
                avg_auc, avg_f1, avg_acc, avg_recall = compute_metrics(
                    # type: ignore
                    outputs, targets, models[model_name]["best_threshold"])[:4]
                models_metrics[model_name][mode][exit]['auc'] = avg_auc
                models_metrics[model_name][mode][exit]['accuracy'] = avg_acc
                models_metrics[model_name][mode][exit]['f1'] = avg_f1
                models_metrics[model_name][mode][exit]['recall'] = avg_recall
                models_metrics[model_name][mode][exit]['entropy'] = float(
                    custom_entropy_formula(outputs.detach().cpu().numpy()))

    print("\n\nResults:")  # count_exit
    print(json.dumps(
        {model: models_metrics[model]["test"] for model in models_metrics}, indent=4))

    # plot
    plt_dir = os.path.join(output_dir, 'plots')
    print(f'\n\nPlots are saved in {plt_dir}\n')
    os.makedirs(plt_dir, exist_ok=True)

    plot_exit_counts(count_exit, plt_dir, save_file=save_plots)

    for mode in ['train', 'val']:
        plot_model_metrics(models_metrics, mode, plt_dir, save_file=save_plots)
        plot_model_metrics2(models_metrics, plt_dir, save_file=save_plots)

    for m in ['accuracy', 'entropy']:
        boxplot_metrics(m, models_metrics, plt_dir, save_file=save_plots)
        barplot_metrics(m, models_metrics, plt_dir, save_file=save_plots)
        all_datas = {el: [] for el in models_metrics}
        for model_name in models_metrics:
            for mode in models_metrics[model_name]:
                for exit in models_metrics[model_name][mode]:
                    if mode in ['test', 'mcd_validation']:
                        all_datas[model_name].append(
                            models_metrics[model_name][mode][exit][m])
                    else:
                        all_datas[model_name] += models_metrics[model_name][mode][exit][m]
            all_datas[model_name] = np.array(
                all_datas[model_name])

        plot_cumulative_count(all_datas,
                              x_label=m.capitalize(), y_label='Number of Samples',
                              title=f'Cumulative Count of {m}', save_file=save_plots,
                              filename=os.path.join(plt_dir, f'cumulative_count_{m}.png'))

    modes = ['test', 'mcd_validation']
    for mode in modes:
        diff_confidence = {model: {exit: compute_diff_and_confidence(
            models[model][mode][exit], mcd=mode == "mcd_validation") for exit in models[model][mode]} for model in models}
        for model in diff_confidence:
            title = f'{mode} - Difference vs Confidence for Exit'
            plot_dictionary_subplots(diff_confidence[model], x_label='Confidence', y_label='Difference', super_title=model, out_dir=plt_dir,
                                     title_prefix=title, save_file=save_plots, filename=f'diff_confidence_{model}_{mode}.png')

    mode = 'mcd_validation'
    diff_var = {model: {exit: compute_diff_and_variational(
        models[model][mode][exit], mcd=mode == "mcd_validation") for exit in models[model][mode]} for model in models}
    for model in diff_var:
        title = f'Error vs Variation ratios for Exit'
        plot_dictionary_subplots(diff_var[model], x_label='Variation ratios', y_label='Difference', super_title=f'{model} - {mode}', out_dir=plt_dir,
                                 title_prefix=title, save_file=save_plots, filename=f'diff_var_{model}_{mode}.png')

    all_metrics = ['accuracy', 'auc', 'f1', 'recall']
    metric_confidence = {model: {metric: []
                                 for metric in all_metrics} for model in models}
    for model in models_metrics:
        for mode in models_metrics[model]:
            for exit in models_metrics[model][mode]:
                for metric in all_metrics:
                    if mode not in ['test', 'mcd_validation']:
                        for i in range(len(models_metrics[model][mode][exit][metric])):
                            metric_confidence[model][metric].append((models_metrics[model][mode][exit][metric][i],
                                                                    1 -
                                                                     models_metrics[model][mode][exit]['entropy'][i],
                                                                    y_lens[mode]))
                    else:
                        metric_confidence[model][metric].append((models_metrics[model][mode][exit][metric],
                                                                 1 -
                                                                 models_metrics[model][mode][exit]['entropy'],
                                                                 y_lens[mode]))

    plot_metric_vs_confidence(metric_confidence, 'accuracy',  min_confidence=0,
                              out_dir=plt_dir, save_file=save_plots)

    modes = ['test', 'mcd_validation']
    metrics = ['accuracy', 'auc', 'f1', 'recall']
    for mode in modes:
        for metric in metrics:
            metric_confidence = {}
            for model in models:
                metric_confidence[model] = {}
                for exit in models[model][mode]:
                    metric_confidence[model][exit] = metrics_per_confidence(
                        models[model][mode][exit], metric=metric, step=0.05, mcd=mode == "mcd_validation", threshold=models[model]["best_threshold"])
            super_title = "Monte Carlo Dropout" if mode == "mcd_validation" else "Test"
            super_title += f' - {metric.capitalize()} on Confidence'
            plot_metric_on_confidence(metric_confidence, metric=metric, file_name=f'{mode}_{metric}', super_title=super_title,
                                      out_dir=plt_dir, save_file=save_plots, min_conf=0.5)

    mode = 'mcd_validation'
    metrics = ['accuracy', 'auc', 'f1', 'recall']
    for metric in metrics:
        metric_confidence = {}
        for model in models:
            metric_confidence[model] = {}
            for exit in models[model][mode]:
                metric_confidence[model][exit] = metrics_per_confidence(models[model][mode][exit], metric=metric, uncertainty="varational_ratio",
                                                                        step=0.05, mcd=True, threshold=models[model]["best_threshold"])
        super_title = "Monte Carlo Dropout" if mode == "mcd_validation" else "Test"
        super_title += f' - {metric.capitalize()} on Confidence'
        super_title += ' - Variational Ratio'
        plot_metric_on_confidence(metric_confidence, metric=metric, file_name=f'{mode}_{metric}_var-ratio', super_title=super_title,
                                  out_dir=plt_dir, save_file=save_plots, min_conf=0.5)

    metric = 'accuracy'
    compare_modes = ('train', 'val', 'test')
    metric_trend = {}
    for model in models_metrics:
        metric_trend[model] = {}
        for exit in models_metrics[model][compare_modes[0]]:
            metric_trend[model][exit] = (models_metrics[model][compare_modes[0]][exit][metric],
                                         models_metrics[model][compare_modes[1]
                                                               ][exit][metric],
                                         models_metrics[model][compare_modes[2]][exit][metric])
    plot_metric_trend(metric_trend, metric=metric, super_title='Train vs Validation',
                      out_dir=plt_dir, save_file=save_plots, file_name='train_vs_val.png')

    print("\n\nDone!")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analysis of all models.')
    parser.add_argument('-o', '--output-dir', type=str,
                        default='../../output/', help='Directory to save the results')
    parser.add_argument('-s', '--save', type=bool,
                        default=True, help='Save the results')
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir[-1] == '/' else args.output_dir + '/'
    main(output_dir, args.save)
