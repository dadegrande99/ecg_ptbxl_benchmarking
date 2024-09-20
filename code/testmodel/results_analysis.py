import numpy as np
from utils.metrics import compute_metrics, compare_tensors, custom_entropy_formula_analysis as custom_entropy_formula, compute_diff_and_confidence
from utils import list_directories, create_tensors_from_dataframe, find_files_with_extension
from utils.plotting import (
    plot_exit_counts, boxplot_metrics, barplot_metrics, plot_cumulative_count,
    plot_model_metrics, plot_model_metrics2, plot_dictionary_subplots, plot_metric_vs_confidence
)
import json
import os

# Warning settings
import warnings
warnings.filterwarnings("ignore")

def main(output_dir:str='../../output/'):
    # configuration for the experiments
    with open(f'{output_dir}config.json', 'r') as f:
        config = json.load(f)
    num_epochs = config['num_epochs']
    ## print configuration
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

    # train
    # models = {el : {'train': {}, 'val': {}, 'test': {}} for el in list_directories(f'{output_dir}/models')}
    models = {el : {'train': {}, 'val': {}, 'test': {}, 'mcd_validation': {}} for el in list_directories(f'{output_dir}/models')}
    models_metrics = {el : {'train': {}, 'val': {}, 'test': {}} for el in list_directories(f'{output_dir}/models')}

    # manage exist
    for model_name in models:
        exits = list_directories(f'{output_dir}/models/{model_name}/{"train"}')
        for mode in models[model_name]:
            models[model_name][mode] = {el : {} for el in range(len(exits))}
            models_metrics[model_name][mode] = {el : {} for el in range(len(exits))}
    count_exit = {}

    # define metrics
    metrics = ['auc', 'accuracy', 'f1', 'entropy', 'recall']
    for model_name in models_metrics:
        for mode in models_metrics[model_name]:
            for exit in models_metrics[model_name][mode]:
                models_metrics[model_name][mode][exit] = {metric: [] for metric in metrics}

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
                    models[model_name][mode][exit][epoch] = create_tensors_from_dataframe(os.path.join(output_dir, 'models', model_name, mode, f'exit_{exit}', el))

        ## metrics
        for model_name in models_metrics:
            for exit in models_metrics[model_name][mode]:
                for i in models[model_name][mode][exit]:
                    outputs, targets = models[model_name][mode][exit][i]
                    avg_auc, avg_f1, avg_acc, avg_recall, aucs, f1_scores, accuracies, recall = compute_metrics(outputs, targets, models[model_name]["best_threshold"]) # type: ignore
                    models_metrics[model_name][mode][exit]['auc'].append(avg_auc)
                    models_metrics[model_name][mode][exit]['accuracy'].append(avg_acc)
                    models_metrics[model_name][mode][exit]['f1'].append(avg_f1)
                    models_metrics[model_name][mode][exit]['recall'].append(avg_recall)
                    models_metrics[model_name][mode][exit]['entropy'].append(float(custom_entropy_formula(outputs.detach().cpu().numpy())))

    # test & mcd validation
    modes = ['test', 'mcd_validation']
    for mode in modes:
        for model_name in models:
            for exit in models[model_name][mode]:
                models[model_name][mode][exit] = create_tensors_from_dataframe(os.path.join(output_dir, 'models', model_name, mode, f'exit_{exit}.csv'))

    ## metrics
    for mode in modes:
        for model_name in models_metrics:
            for exit in models_metrics[model_name][mode]:
                outputs, targets = models[model_name][mode][exit]
                if mode == 'mcd_validation':
                    outputs = outputs.mean(2)
                avg_auc, avg_f1, avg_acc, avg_recall, aucs, f1_scores, accuracies, recall = compute_metrics(outputs, targets, models[model_name]["best_threshold"]) # type: ignore
                models_metrics[model_name][mode][exit]['auc'] = avg_auc
                models_metrics[model_name][mode][exit]['accuracy'] = avg_acc
                models_metrics[model_name][mode][exit]['f1'] = avg_f1
                models_metrics[model_name][mode][exit]['recall'] = avg_recall
                models_metrics[model_name][mode][exit]['entropy'] = float(custom_entropy_formula(outputs.detach().cpu().numpy()))
    
    print("\n\nResults:")#count_exit
    print(json.dumps({model:models_metrics[model]["test"] for model in models_metrics}, indent=4))

    # plot
    plt_dir = os.path.join(output_dir, 'plots')
    print(f'\n\nPlots are saved in {plt_dir}\n')
    os.makedirs(plt_dir, exist_ok=True)

    plot_exit_counts(count_exit, plt_dir)

    for mode in ['train', 'val']:
        plot_model_metrics(models_metrics, mode, plt_dir)
        plot_model_metrics2(models_metrics, plt_dir)

    
    for m in ['accuracy', 'entropy']:
        boxplot_metrics(m, models_metrics, plt_dir)
        barplot_metrics(m, models_metrics, plt_dir)
        all_datas = { el : [] for el in models_metrics}
        for model_name in models_metrics:
            for mode in models_metrics[model_name]:
                for exit in models_metrics[model_name][mode]:
                    if mode in ['test', 'mcd_validation']:
                        all_datas[model_name].append(models_metrics[model_name][mode][exit][m])
                    else:
                        all_datas[model_name] += models_metrics[model_name][mode][exit][m]
            all_datas[model_name] = np.array(all_datas[model_name]) # type: ignore
        
        
        plot_cumulative_count(all_datas, # type: ignore
                              x_label=m.capitalize(), y_label='Number of Samples',
                              title=f'Cumulative Count of {m}',
                              save_file=True, filename=os.path.join(plt_dir, f'cumulative_count_{m}.png'))


    mode = 'test'
    diff_confidence = {model: {exit: compute_diff_and_confidence(models[model][mode][exit]) for exit in models[model][mode]} for model in models}
    for model in diff_confidence:
        plot_dictionary_subplots(diff_confidence[model], x_label='Confidence', y_label='Difference', super_title=model, out_dir=plt_dir,
                             title_prefix='Difference vs Confidence for Exit', save_file=True)
        
    all_metrics = ['accuracy', 'auc', 'f1', 'recall']
    metric_confidence = {model: {metric : [] for metric in all_metrics} for model in models}
    for model in models_metrics:
        for mode in models_metrics[model]:
            for exit in models_metrics[model][mode]:
                for metric in all_metrics:
                    if mode not in ['test', 'mcd_validation']:
                        for i in range(len(models_metrics[model][mode][exit][metric])):
                            metric_confidence[model][metric].append((models_metrics[model][mode][exit][metric][i], 
                                                                    1 - models_metrics[model][mode][exit]['entropy'][i], 
                                                                    y_lens[mode]))
                    else:
                        metric_confidence[model][metric].append((models_metrics[model][mode][exit][metric], 
                                                            1 - models_metrics[model][mode][exit]['entropy'], 
                                                            y_lens[mode]))
                        
    plot_metric_vs_confidence(metric_confidence, 'accuracy', out_dir=plt_dir)
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analysis of all models.')
    parser.add_argument('-o', '--output-dir', type=str, default='../../output/', help='Directory to save the results')
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir[-1] == '/' else args.output_dir + '/'
    main(output_dir)