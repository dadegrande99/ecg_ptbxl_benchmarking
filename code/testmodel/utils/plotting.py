import matplotlib.pyplot as plt

def plot_model_metrics(models_metrics, mode):
    for model in models_metrics:
        plt.figure(figsize=(14, 8))
        for metric in models_metrics[model][mode]:
            if metric == 'entropy':
                continue
            plt.plot(range(1, len(models_metrics[model][mode][metric])+1), 
                    models_metrics[model][mode][metric], 
                    label=f'{metric} - Test {models_metrics[model]["test"][metric]:.2f}')
   
        # Add titles, labels, legends, and grid
        plt.title(f'Metrics Over Epochs for {model} - {mode}')
        plt.xlabel('Epochs')
        plt.xticks(range(1, len(models_metrics[model][mode][metric])+1))
        plt.ylabel('Metric Values')
        plt.legend()
        plt.grid(True)
        return plt.show()


def plot_model_metrics2(models_metrics):
    for model in models_metrics:
        plt.figure(figsize=(14, 8))
        for metric in models_metrics[model]["train"]:
            if metric == 'entropy':
                continue
            mode = 'train'
            line1, = plt.plot(range(1, len(models_metrics[model][mode][metric])+1), 
                    models_metrics[model][mode][metric], linestyle='--')
            mode = 'val'
            plt.plot(range(1, len(models_metrics[model][mode][metric])+1), 
                    models_metrics[model][mode][metric], color=line1.get_color(),
                    label=f'{metric} - Test {models_metrics[model]['test'][metric]:.2f}')
            
        # Add titles, labels, legends, and grid
        plt.title(f'Metrics Over Epochs for {model}')
        plt.xlabel('Epochs')
        plt.xticks(range(1, len(models_metrics[model][mode][metric])+1))
        plt.ylabel('Metric Values')
        plt.legend()
        plt.grid(True)
        return plt.show()
    

def compare_models_on_metric(models_metrics, metric_to_compare):
    plt.figure(figsize=(14, 8))
    for model in models_metrics:
        mode = 'train'
        line1, = plt.plot(range(1, len(models_metrics[model][mode][metric_to_compare])+1), 
                models_metrics[model][mode][metric_to_compare], linestyle='--')
        mode = 'val'
        plt.plot(range(1, len(models_metrics[model][mode][metric_to_compare])+1), 
                models_metrics[model][mode][metric_to_compare], color=line1.get_color(),
                label=f'{model} - Test: {models_metrics[model]['test'][metric_to_compare]:.2f}')
    
    # Add titles, labels, legends, and grid
    plt.title(f'Comparison of Models on {metric_to_compare}')
    plt.xlabel('Epochs')
    plt.xticks(range(1, len(models_metrics[model][mode][metric_to_compare])+1))
    plt.ylabel('Metric Values')
    plt.legend()
    plt.grid(True)
    return plt.show()

# compare models on 2 metrics
def compare_models_on_metrics(models_metrics, metric1, metric2):
    plt.figure(figsize=(14, 8))
    for model in models_metrics:
        mode = 'train'
        line1, = plt.plot(range(1, len(models_metrics[model][mode][metric1])+1), 
                models_metrics[model][mode][metric1], linestyle='--')
        line2, = plt.plot(range(1, len(models_metrics[model][mode][metric2])+1), 
                models_metrics[model][mode][metric2], linestyle='--')
        mode = 'val'
        plt.plot(range(1, len(models_metrics[model][mode][metric1])+1), 
                models_metrics[model][mode][metric1], color=line1.get_color(),
                label=f'{metric1} - Test: {models_metrics[model]['test'][metric1]:.2f}')
        plt.plot(range(1, len(models_metrics[model][mode][metric2])+1),
                models_metrics[model][mode][metric2], color=line2.get_color(),
                label=f'{metric2} - Test: {models_metrics[model]['test'][metric2]:.2f}')
    
    # Add titles, labels, legends, and grid
    plt.title(f'Comparison of Models on {metric1} and {metric2}')
    plt.xlabel('Epochs')
    plt.xticks(range(1, len(models_metrics[model][mode][metric1])+1))
    plt.ylabel('Metric Values')
    plt.legend()
    plt.grid(True)
    return plt.show()
