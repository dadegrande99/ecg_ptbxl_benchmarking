import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.metrics import compute_metrics, custom_entropy_formula
from utils import list_directories, create_tensors_from_dataframe, find_files_with_extension
import json
import torch
from typing import Dict, Optional
import os

# Warning settings
import warnings
#warnings.filterwarnings("ignore")

# Plot settings
plt.style.use([s for s in plt.style.available if 'whitegrid' in s][0])
plt.rcParams['figure.figsize'] = [16, 9]
plt.rcParams['figure.dpi'] = 100


# Function to plot ROC curve
def plot_roc_curve(fpr, tpr, roc_auc, title):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def plot_exit_counts(count_exit, out_dir):
    models = list(count_exit.keys())

    max_exits = max(len(v) for v in count_exit.values())

    exits_data = {str(k): [0]*len(models) for k in range(max_exits)}

    for model_idx, (model, counts) in enumerate(count_exit.items()):
        for exit_stage, count in counts.items():
            exits_data[exit_stage][model_idx] = count

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.15
    x = np.arange(len(models))

    for i, (stage, counts) in enumerate(exits_data.items()):
        ax.bar(x + i * bar_width, counts, width=bar_width, label=f'Exit stage {stage}')

    ax.set_xlabel('Models')
    ax.set_ylabel('Exit Counts')
    ax.set_title('Exit Counts by Model and Stage')
    ax.set_xticks(x + bar_width * (max_exits - 1) / 2)
    ax.set_xticklabels(models)
    ax.legend()

    plt.savefig(os.path.join(out_dir, 'exit_counts.png'))
    plt.close()


def plot_cumulative_count(data_dict, min_threshold=None, max_threshold=None,
                          x_label='Threshold', y_label='Number of Samples',
                          title='Progressive Cumulative Count Above Each Threshold',
                          grid=True, save_file=True, filename='cumulative_count.png'):
    """
    Plots a progressive cumulative count of the number of data points exceeding each unique threshold
    for multiple models, using a common minimum threshold for all models.

    Parameters:
    - data_dict: dictionary with model names as keys and arrays of metric values as values.
    - min_threshold: minimum value to start the threshold (inclusive), calculated from all models if not specified.
    - max_threshold: maximum value to end the threshold (inclusive).
    - x_label: label for the x-axis.
    - y_label: label for the y-axis.
    - title: title of the plot.
    - grid: whether to display grid lines on the plot.
    - save_file: if True, saves the plot to a file; otherwise, displays the plot.
    - filename: name of the file to save the plot to.
    """
    if min_threshold is None or max_threshold is None:
        all_data = np.concatenate(list(data_dict.values()))
        if min_threshold is None:
            min_threshold = np.min(all_data)
        if max_threshold is None:
            max_threshold = np.max(all_data)
    
    plt.figure(figsize=(10, 6))
    # Plot data for each model using the global min and max threshold
    for model_name, data in data_dict.items():
        thresholds = np.linspace(min_threshold, max_threshold, 100)
        counts = [np.sum(data > threshold) for threshold in thresholds]
        
        plt.plot(thresholds, counts, label=model_name)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(grid)
    plt.legend()

    # Save the plot or display it
    if save_file:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def boxplot_metrics(metric: str, models_metrics: dict, out_dir: str) -> None:
    """
    Creates and saves a boxplot for the specified metric across different models, categorized by exits.

    Parameters:
    - metric (str): The metric to plot.
    - models_metrics (dict): Nested dictionary with model names as keys at the top level,
                             and 'phases' dict which further maps to 'exits' dict containing metric values.
    - out_dir (str): Directory where the boxplot image will be saved.
    """
    # Collecting data for DataFrame
    data = []
    for model_name, phases in models_metrics.items():
        for phase, exits in phases.items():
            for exit_number, metrics in exits.items():
                # Ensure the specific metric exists and handle it as a list
                metric_values = metrics.get(metric)
                if metric_values is not None:
                    # Normalize single float values to list
                    if isinstance(metric_values, float):
                        metric_values = [metric_values]
                    for value in metric_values:
                        data.append({
                            'Model': model_name,
                            'Exit': f'Exit {exit_number}',  # Labeling the exit
                            'Value': value
                        })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        print(f'No data available for {metric} to plot.')
        return

    # Creating the boxplot using seaborn for easy and attractive visualization
    plt.figure(figsize=(12, 6))
    x_val = 'Model'
    hue_val = 'Exit'
    sns.boxplot(x=x_val, y='Value', hue=hue_val, data=df)
    plt.title(f'Boxplot of {metric} by Exit and Model')
    plt.xlabel(x_val)
    plt.ylabel(metric.capitalize())
    # plt.legend(title=hue_val)
        
    # Save the plot to the specified directory
    plot_file = os.path.join(out_dir, f'{metric}_boxplot.png')
    plt.savefig(plot_file)
    plt.close()
    print(f'Boxplot saved to {plot_file}')

def barplot_metrics(metric: str, models_metrics: dict, out_dir: str) -> None:
    """
    
    """

    # Collecting data for DataFrame
    data = []
    phase = 'test'

    for model_name in models_metrics:
        for exit in models_metrics[model_name][phase]:
            metric_values = models_metrics[model_name][phase][exit].get(metric)
            if metric_values is not None:
                data.append({
                    'Model': model_name,
                    'Exit': f'Exit {exit}',  # Labeling the exit
                    'Value': metric_values
                })

    """for model_name, phases in models_metrics.items():
        for exit_number, metrics in model_name[phase].items():
            # Ensure the specific metric exists and handle it as a list
            metric_values = metrics.get(metric)
            if metric_values is not None:
                # Normalize single float values to list
                if isinstance(metric_values, float):
                    metric_values = [metric_values]
                for value in metric_values:
                    data.append({
                        'Model': model_name,
                        'Exit': f'Exit {exit_number}',  # Labeling the exit
                        'Value': value
                    })"""
    
    df = pd.DataFrame(data)
    
    if df.empty:
        print(f'No data available for {metric} to plot.')
        return

    # Creating the boxplot using seaborn for easy and attractive visualization
    plt.figure(figsize=(12, 6))
    x_val = 'Model'
    hue_val = 'Exit'
    sns.barplot(x=x_val, y='Value', hue=hue_val, data=df)
    plt.title(f'Barplot of {metric} by Exit and Model')
    plt.xlabel(x_val)
    plt.ylabel(metric.capitalize())
    # plt.legend(title=hue_val)
        
    # Save the plot to the specified directory
    plot_file = os.path.join(out_dir, f'{metric}_barplot.png')
    plt.savefig(plot_file)
    plt.close()
    print(f'Barplot saved to {plot_file}')


def plot_accuracy_with_confidence(model_data, x_label='Tau', y_label='Accuracy on Examples', 
                                  title='Model Accuracy as a Function of Tau', out_dir='./'):
    """
    Plots the accuracy and confidence intervals for different models as a function of tau.
    
    Parameters:
    - model_data: Dictionary with model names as keys and tuples of (tau_values, accuracies, lower_bounds, upper_bounds) as values.
    - x_label: Label for the x-axis.
    - y_label: Label for the y-axis.
    - title: Title of the plot.
    - out_dir: Directory where the plot will be saved.
    """
    plt.figure(figsize=(10, 6))
    for model_name, (tau_values, accuracies, lower_bounds, upper_bounds) in model_data.items():
        plt.plot(tau_values, accuracies, label=model_name)
        plt.fill_between(tau_values, lower_bounds, upper_bounds, alpha=0.2)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_dir}/{title.replace(' ', '_').lower()}.png")
    plt.show()


def custom_entropy_formula(predictions):
    # Ensure predictions are within the valid range for logarithms
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
    
    # If predictions are of shape (n, 1), assume binary classification and calculate the complementary probabilities
    if predictions.shape[1] == 1:
        complementary_predictions = 1 - predictions
        predictions = np.hstack((complementary_predictions, predictions))

    # Normalize predictions to ensure they sum to 1 (if necessary)
    predictions /= np.sum(predictions, axis=1, keepdims=True)
    
    # Calculate entropy
    entropy = -np.sum(predictions * np.log(predictions), axis=1)

    # Since this is a binary classification, max entropy is log(2)
    max_entropy = np.log(2)
    normalized_entropy = np.mean(entropy) / max_entropy

    return normalized_entropy


# retrieve datas from csv
def create_tensors_from_dataframe(csv_path, output_prefix='output_', target_prefix='target_'):
    # Load the dataframe
    df = pd.read_csv(csv_path)

    # Get the output and target columns
    output_columns = [col for col in df.columns if col.startswith(output_prefix)]
    target_columns = [col for col in df.columns if col.startswith(target_prefix)]
    
    # Create the tensors
    outputs_tensor = torch.tensor(df[output_columns].to_numpy())
    targets_tensor = torch.tensor(df[target_columns].to_numpy())
    
    return outputs_tensor, targets_tensor

import torch

def compare_tensors(tensor_pair1, tensor_pair2):
    # retrieve pairs
    outputs_tensor1, targets_tensor1 = tensor_pair1
    outputs_tensor2, targets_tensor2 = tensor_pair2
    
    # compare output & target
    outputs_equal = torch.equal(outputs_tensor1, outputs_tensor2)
    targets_equal = torch.equal(targets_tensor1, targets_tensor2)
    
    return outputs_equal and targets_equal


def main(output_dir='../../output/'):
    # configuration for the experiments
    with open(f'{output_dir}config.json', 'r') as f:
        config = json.load(f)
    num_epochs = config['num_epochs']
    ## print configuration
    print("Configuration for the experiments:")
    max_key_len = max([len(k) for k in config.keys()])
    for key, value in config.items():
        print(f'- {key}:{(max_key_len - len(key) + 1) * " "} {value}')

    print("\n\nModels:")
    for el in list_directories(f'{output_dir}/models'):
        print(f'- {el}')

    # train
    models = {el : {'train': {}, 'val': {}, 'test': {}} for el in list_directories(f'{output_dir}/models')}
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
                    try:
                        models[model_name][mode][exit][epoch] = create_tensors_from_dataframe(os.path.join(output_dir, 'models',
                                                                                                           model_name, mode, f'exit_{exit}', el))
                    except:
                        continue

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

    # test
    mode = 'test'
    for model_name in models:
        for exit in models[model_name][mode]:
            try:
                models[model_name][mode][exit] = create_tensors_from_dataframe(os.path.join(output_dir, 'models', model_name, mode, f'exit_{exit}.csv'))
            except:
                continue
    ## test metrics
    for model_name in models_metrics:
        for exit in models_metrics[model_name][mode]:
            outputs, targets = models[model_name][mode][exit]
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
    os.makedirs(plt_dir, exist_ok=True)
    plot_exit_counts(count_exit, plt_dir)

    
    for m in ['accuracy', 'entropy']:
        boxplot_metrics(m, models_metrics, plt_dir)
        barplot_metrics(m, models_metrics, plt_dir)
        all_datas = { el : [] for el in models_metrics}
        for model_name in models_metrics:
            for mode in models_metrics[model_name]:
                for exit in models_metrics[model_name][mode]:
                    if mode == 'test':
                        all_datas[model_name].append(models_metrics[model_name][mode][exit][m])
                    else:
                        all_datas[model_name] += models_metrics[model_name][mode][exit][m]
            all_datas[model_name] = np.array(all_datas[model_name]) # type: ignore
        
        
        plot_cumulative_count(all_datas, # type: ignore
                              x_label=m.capitalize(), y_label='Number of Samples',
                              title=f'Cumulative Count of {m}',
                              save_file=True, filename=os.path.join(plt_dir, f'cumulative_count_{m}.png'))      
        
        
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analysis of all models.')
    parser.add_argument('-o', '--output-dir', type=str, default='../../output/', help='Directory to save the results')
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir[-1] == '/' else args.output_dir + '/'
    output_dir = "/Users/davidegrandesso/Desktop/output/"
    output_dir = "/Users/davidegrandesso/Desktop/output copy/"
    main(output_dir)