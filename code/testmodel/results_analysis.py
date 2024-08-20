import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.metrics import compute_metrics, custom_entropy_formula
from utils import list_directories, create_tensors_from_dataframe, find_files_with_extension
import json
import torch
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


def boxplot_metrics(metric, models_metrics, out_dir):
    # Creazione di un DataFrame per contenere tutti i dati necessari per il boxplot
    data = []
    for model_name, phases in models_metrics.items():
        for phase, exits in phases.items():
            for exit_number, metrics in exits.items():
                # Assicurati che il valore metrico sia una lista prima di iterare
                metric_values = metrics.get(metric)
                if metric_values is not None:
                    if isinstance(metric_values, float):  # Se è un float, convertilo in lista
                        metric_values = [metric_values]
                    for value in metric_values:
                        data.append({
                            'Model': model_name,
                            'Phase': phase,
                            'Exit': exit_number,
                            'Value': value
                        })
    
    df = pd.DataFrame(data)
    
    if not df.empty:
        # Creazione del boxplot utilizzando seaborn per una visualizzazione facile e attraente
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Model', y='Value', hue='Phase', data=df)
        plt.title(f'Boxplot of {metric} by Model and Phase')
        plt.xlabel('Model')
        plt.ylabel(metric.capitalize())
        plt.legend(title='Phase')
        
        # Salvataggio del plot nel directory specificata
        plot_file = os.path.join(out_dir, f'{metric}_boxplot.png')
        plt.savefig(plot_file)
        plt.close()
        
        print(f'Boxplot saved to {plot_file}')
    else:
        print(f'No data available for {metric} to plot.')





def custom_entropy_formula(predictions: np.array): # type: ignore
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)  # Ensure values are in the range [1e-9, 1-1e-9]
    predictions = np.mean(predictions, axis=0)  # Average predictions across batch
    
    entropy = -np.nansum(predictions * np.log(predictions), axis=0) / np.log(predictions.shape[0])
    
    return entropy

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
    metrics = ['auc', 'accuracy', 'f1', 'entropy']
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
                max_exit = -1
                # print( output_dir, 'models', model_name, mode, f'exit_{exit}')
                # print(type(exit))
                # print(find_files_with_extension(os.path.join(output_dir, 'models', model_name, mode, f'exit_{exit}'), 'csv'), '\n')
                for el in find_files_with_extension(os.path.join(output_dir, 'models', model_name, mode, f'exit_{exit}'), 'csv'):
                    #print(el)
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
                    try:
                        outputs, targets = models[model_name][mode][exit][i]
                        avg_auc, avg_f1, avg_acc, aucs, f1_scores, accuracies = compute_metrics(outputs, targets, models[model_name]["best_threshold"]) # type: ignore
                        models_metrics[model_name][mode][exit]['auc'].append(avg_auc)
                        models_metrics[model_name][mode][exit]['accuracy'].append(avg_acc)
                        models_metrics[model_name][mode][exit]['f1'].append(avg_f1)
                        models_metrics[model_name][mode][exit]['entropy'].append(float(custom_entropy_formula(outputs / outputs.sum(dim=1, keepdim=True))))
                    except:
                        continue

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
            avg_auc, avg_f1, avg_acc, aucs, f1_scores, accuracies = compute_metrics(outputs, targets, models[model_name]["best_threshold"]) # type: ignore
            models_metrics[model_name][mode][exit]['auc'] = avg_auc
            models_metrics[model_name][mode][exit]['accuracy'] = avg_acc
            models_metrics[model_name][mode][exit]['f1'] = avg_f1
            models_metrics[model_name][mode][exit]['entropy'] = float(custom_entropy_formula(outputs.detach().cpu().numpy()))
    


    print("\n\nResults:")#count_exit
    print(models)
    # print(json.dumps(models_metrics, indent=4))
    print(json.dumps(count_exit, indent=4))

    # plot
    plt_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plt_dir, exist_ok=True)
    plot_exit_counts(count_exit, plt_dir)

    
    for m in ['accuracy', 'entropy']:
        boxplot_metrics(m, models_metrics, plt_dir)



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analysis of all models.')
    parser.add_argument('-o', '--output-dir', type=str, default='../../output/', help='Directory to save the results')
    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir[-1] == '/' else args.output_dir + '/'
    main(output_dir)