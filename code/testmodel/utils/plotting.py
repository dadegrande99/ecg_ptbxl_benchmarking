import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
import os

def plot_model_metrics(models_metrics, mode, out_dir:str="",
                       figsize: tuple = (14, 8), dpi: int = 100,
                       x_label: str = 'Epochs', y_label: str = 'Metric Values',
                       title_prefix: str = 'Metrics Over Epochs for', grid: bool = True,
                       legend:bool=True, save_file:bool=True) -> None:
    """
    Generates and displays line plots of various metrics over epochs for each model in `models_metrics`.

    Parameters:
    - models_metrics (dict): A nested dictionary containing metric values for each model
    - mode (str): The mode for which metrics are to be plotted (e.g., 'train', 'validation')
    - out_dir (str): Directory where the plots will be saved. Default is the current working directory
    - figsize (tuple): Figure size in inches as a tuple (width, height). Default is (14, 8)
    - dpi (int): Dots per inch (resolution) of the figure. Default is 100
    - x_label (str): Label for the x-axis. Default is 'Epochs'
    - y_label (str): Label for the y-axis. Default is 'Metric Values'
    - title_prefix (str): Prefix for the plot title. Default is 'Metrics Over Epochs for'
    - grid (bool): Whether to display a grid on the plot. Default is True
    - legend (bool): Whether to display a legend on the plot. Default is True
    - save_file (bool): If True, saves the plot to a file; if False, displays the plot. Default is True

    Returns:
    - None
    """

    for model in models_metrics:
        plt.figure(figsize=figsize, dpi=dpi)
        for exit in models_metrics[model][mode]:
            for metric in models_metrics[model][mode][exit]:
                if metric == 'entropy':
                    continue
                plt.plot(
                    range(1, len(models_metrics[model][mode][exit][metric]) + 1),
                    models_metrics[model][mode][exit][metric],
                    label=f'{metric} - Exit {exit}'
                )

        # Add titles, labels, legends, and grid
        title = f'{title_prefix} {model} ({mode.capitalize()})'
        filename = f'{model}_{mode}_metrics.png'
        plt.title(title)
        plt.xlabel(x_label)
        plt.xticks(range(1, len(models_metrics[model][mode][exit][metric]) + 1))
        plt.ylabel(y_label)
        plt.grid(grid)
        if legend:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.subplots_adjust(right=0.8)
        else:
            plt.legend().remove()
        if save_file:
            plt.savefig(os.path.join(out_dir, filename))
            print(f'Plot \033[3m{title}\033[0m saved to {filename}')
        else:
            plt.show()
        plt.close()


def plot_model_metrics2(models_metrics, out_dir:str="",
                        x_label: str = 'Epochs', y_label: str = 'Metric Values',
                        title_prefix: str = 'Metrics Over Epochs for', grid: bool = True,
                        figsize: tuple = (14, 8), dpi: int = 100,
                        legend:bool=True, save_file:bool=True) -> None:
    """
    Generates and displays line plots comparing training and validation metrics over epochs for each model in `models_metrics`

    Parameters:
    - models_metrics (dict): A nested dictionary containing metric values for each model
    - out_dir (str): Directory where the plots will be saved. Default is the current working directory
    - x_label (str): Label for the x-axis. Default is 'Epochs'
    - y_label (str): Label for the y-axis. Default is 'Metric Values'
    - title_prefix (str): Prefix for the plot title. Default is 'Metrics Over Epochs for'
    - grid (bool): Whether to display a grid on the plot. Default is True
    - figsize (tuple): Figure size in inches as a tuple (width, height). Default is (14, 8)
    - dpi (int): Dots per inch (resolution) of the figure. Default is 100
    - legend (bool): Whether to display a legend on the plot. Default is True
    - save_file (bool): If True, saves the plot to a file; if False, displays the plot. Default is True

    Returns:
    - None
    """

    for model in models_metrics:
        plt.figure(figsize=figsize, dpi=dpi)
        for exit in models_metrics[model]["train"]:
            for metric in models_metrics[model]["train"][exit]:
                if metric == 'entropy':
                    continue
                # Plot training metric with dashed line
                mode = 'train'
                line1, = plt.plot(
                    range(1, len(models_metrics[model][mode][exit][metric]) + 1),
                    models_metrics[model][mode][exit][metric],
                    linestyle='--'
                )
                # Plot validation metric with the same color as training metric
                mode = 'val'
                plt.plot(
                    range(1, len(models_metrics[model][mode][exit][metric]) + 1),
                    models_metrics[model][mode][exit][metric],
                    color=line1.get_color(),
                    label=f'{metric}{exit} - Test {models_metrics[model]["test"][exit][metric]:.2f}'
                )
        
        filename = f'{model}_train_val_metrics.png'
        title = f'{title_prefix} {model} (Train vs Validation)'
        
        # Add titles, labels, legends, and grid
        plt.xlabel(x_label)
        plt.xticks(range(1, len(models_metrics[model][mode][exit][metric]) + 1))
        plt.ylabel(y_label)
        plt.grid(grid)
        plt.title(title)
        if legend:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.subplots_adjust(right=0.8)
        else:
            plt.legend().remove()
        if save_file:
            plt.savefig(os.path.join(out_dir, filename))
            print(f'Plot \033[3m{title}\033[0m saved to {filename}')
        else:
            plt.show()
        plt.close()
    

def compare_models_on_metric(models_metrics, metric_to_compare: str,
                             x_label: str = 'Epochs', y_label: str = 'Metric Values',
                             title_prefix: str = 'Comparison of Models on', grid: bool = True,
                             figsize: tuple = (14, 8), dpi: int = 100) -> None:
    """
    Generates and displays line plots comparing multiple models on a specific metric over epochs

    Parameters:
    - models_metrics (dict): Nested dictionary containing metric values for each model
    - metric_to_compare (str): The name of the metric to compare across models
    - x_label (str): Label for the x-axis. Default is 'Epochs'
    - y_label (str): Label for the y-axis. Default is 'Metric Values'
    - title_prefix (str): Prefix for the plot title. Default is 'Comparison of Models on'
    - grid (bool): Whether to display a grid on the plot. Default is True
    - figsize (tuple): Figure size in inches as a tuple (width, height). Default is (14, 8)
    - dpi (int): Dots per inch (resolution) of the figure. Default is 100

    Returns:
    - None
    """

    plt.figure(figsize=figsize, dpi=dpi)
    max_epochs = 0  # To determine the maximum number of epochs across models
    for model in models_metrics:
        # Plot training metric with dashed line
        train_epochs = len(models_metrics[model]['train'][metric_to_compare])
        if train_epochs > max_epochs:
            max_epochs = train_epochs
        line1, = plt.plot(
            range(1, train_epochs + 1),
            models_metrics[model]['train'][metric_to_compare],
            linestyle='--'
        )
        # Plot validation metric with the same color as training metric
        val_epochs = len(models_metrics[model]['val'][metric_to_compare])
        plt.plot(
            range(1, val_epochs + 1),
            models_metrics[model]['val'][metric_to_compare],
            color=line1.get_color(),
            label=f"{model} - Test: {models_metrics[model]['test'][metric_to_compare]:.2f}"
        )

    # Add titles, labels, legends, and grid
    plt.title(f'{title_prefix} {metric_to_compare}')
    plt.xlabel(x_label)
    plt.xticks(range(1, max_epochs + 1))
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(grid)
    plt.show()


def compare_models_on_metrics(models_metrics, metric1: str, metric2: str,
                              x_label: str = 'Epochs', y_label: str = 'Metric Values',
                              title_prefix: str = 'Comparison of Models on', grid: bool = True,
                              figsize: tuple = (14, 8), dpi: int = 100) -> None:
    """
    Generates and displays line plots comparing multiple models on two specific metrics over epochs

    Parameters:
    - models_metrics (dict): Nested dictionary containing metric values for each model
    - metric1 (str): The first metric to compare across models
    - metric2 (str): The second metric to compare across models
    - x_label (str): Label for the x-axis. Default is 'Epochs'
    - y_label (str): Label for the y-axis. Default is 'Metric Values'
    - title_prefix (str): Prefix for the plot title. Default is 'Comparison of Models on'
    - grid (bool): Whether to display a grid on the plot. Default is True
    - figsize (tuple): Figure size in inches as a tuple (width, height). Default is (14, 8)
    - dpi (int): Dots per inch (resolution) of the figure. Default is 100

    Returns:
    - None
    """

    plt.figure(figsize=figsize, dpi=dpi)
    max_epochs = 0  # To determine the maximum number of epochs across models

    # Get a color map with enough distinct colors for the models
    model_list = list(models_metrics.keys())
    num_models = len(model_list)
    colors = cm.get_cmap('tab10', num_models)

    for idx, model in enumerate(models_metrics):
        color = colors(idx)
        # Plot training metrics with dashed lines
        mode = 'train'
        # Plot metric1
        data_metric1 = models_metrics[model][mode][metric1]
        epochs_metric1 = len(data_metric1)
        if epochs_metric1 > max_epochs:
            max_epochs = epochs_metric1
        plt.plot(
            range(1, epochs_metric1 + 1),
            data_metric1,
            linestyle='--',
            color=color,
            label=f"{model} - {metric1} ({mode})"
        )

        # Plot metric2
        data_metric2 = models_metrics[model][mode][metric2]
        epochs_metric2 = len(data_metric2)
        if epochs_metric2 > max_epochs:
            max_epochs = epochs_metric2
        plt.plot(
            range(1, epochs_metric2 + 1),
            data_metric2,
            linestyle='--',
            color=color,
            label=f"{model} - {metric2} ({mode})"
        )

        # Plot validation metrics with solid lines
        mode = 'val'
        # Plot metric1
        data_metric1_val = models_metrics[model][mode][metric1]
        epochs_metric1_val = len(data_metric1_val)
        if epochs_metric1_val > max_epochs:
            max_epochs = epochs_metric1_val
        plt.plot(
            range(1, epochs_metric1_val + 1),
            data_metric1_val,
            linestyle='-',
            color=color,
            label=f"{model} - {metric1} ({mode}) - Test: {models_metrics[model]['test'][metric1]:.2f}"
        )

        # Plot metric2
        data_metric2_val = models_metrics[model][mode][metric2]
        epochs_metric2_val = len(data_metric2_val)
        if epochs_metric2_val > max_epochs:
            max_epochs = epochs_metric2_val
        plt.plot(
            range(1, epochs_metric2_val + 1),
            data_metric2_val,
            linestyle='-',
            color=color,
            label=f"{model} - {metric2} ({mode}) - Test: {models_metrics[model]['test'][metric2]:.2f}"
        )

    # Add titles, labels, legends, and grid
    plt.title(f'{title_prefix} {metric1} and {metric2}')
    plt.xlabel(x_label)
    plt.xticks(range(1, max_epochs + 1))
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(grid)
    plt.show()


def plot_exit_counts(count_exit, out_dir:str,
                     x_label:str='Models', y_label:str='Exit Counts',
                     title:str='Exit Counts by Model and Stage',
                     fig_size:tuple=(10, 6), dpi:int=100,
                     grid:bool=True, save_file:bool=True, filename:str='exit_counts.png') -> None:

    """
    Generates and saves a bar plot showing exit counts for different models and stages.

    Parameters:
    - count_exit (dict): A dictionary with model names as keys and dictionaries as values
                         Each inner dictionary maps exit stages to their respective counts
                         Example: {'model1': {'0': count0, '1': count1}, 'model2': {...}, ...}
    - out_dir (str): Directory where the bar plot image will be saved
    - x_label (str): Label for the x-axis. Default is 'Models'
    - y_label (str): Label for the y-axis. Default is 'Exit Counts'
    - title (str): Title of the plot. Default is 'Exit Counts by Model and Stage'
    - fig_size (tuple): Size of the figure in inches. Default is (10, 6)
    - dpi (int): Resolution of the figure in dots per inch. Default is 100
    - grid (bool): Whether to display a grid on the plot. Default is True
    - save_file (bool): If True, saves the plot to a file; if False, displays the plot. Default is True
    - filename (str): Name of the file to save the plot. Default is 'exit_counts.png'

    Returns:
    - None
    """

    models = list(count_exit.keys())
    max_exits = max(len(v) for v in count_exit.values())
    exits_data = {str(k): [0]*len(models) for k in range(max_exits)}

    for model_idx, (model, counts) in enumerate(count_exit.items()):
        for exit_stage, count in counts.items():
            exits_data[exit_stage][model_idx] = count

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    bar_width = 0.15
    x = np.arange(len(models))

    for i, (stage, counts) in enumerate(exits_data.items()):
        ax.bar(x + i * bar_width, counts, width=bar_width, label=f'Exit {stage}')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(x + bar_width * (max_exits - 1) / 2)
    ax.set_xticklabels(models)
    ax.grid(grid)
    ax.legend()

    if save_file:
        plt.savefig(os.path.join(out_dir, filename))
        print(f'Plot \033[3m{title}\033[0m saved to {filename}')
    else:
        plt.show()

    plt.close()


def boxplot_metrics(metric: str, models_metrics: dict, out_dir: str,
                    x_label: str = 'Model', hue_val: str = "Exit",
                    grid: bool = True, legend: bool = False,
                    figsize: tuple = (12, 6), dpi: int = 100,
                    save_file: bool = True) -> None:
    """
    Creates and saves a boxplot for the specified metric across different models, categorized by exits.

    Parameters:
    - metric (str): The metric to plot
    - models_metrics (dict): Nested dictionary with model names as keys at the top level, and 'phases' dict which further maps to 'exits' dict containing metric values
    - out_dir (str): Directory where the boxplot image will be saved
    - x_label (str): Label for the x-axis. Default is 'Model'
    - hue_val (str): Label for the hue (color) of the plot. Default is 'Exit'
    - grid (bool): Whether to display a grid on the plot. Default is True
    - legend (bool): Whether to display a legend on the plot. Default is False
    - figsize (tuple): Figure size in inches as a tuple (width, height). Default is (12, 6)
    - dpi (int): Dots per inch (resolution) of the figure. Default is 100
    - save_file (bool): If True, saves the plot to a file; if False, displays the plot. Default is True

    Returns:
    - None
    """
    # Collecting data for DataFrame
    data = []
    x_label = "Model"
    hue_val = "Exit"
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
                            x_label: model_name,
                            hue_val: f'Exit {exit_number}',  # Labeling the exit
                            'Value': value
                        })
    
    df = pd.DataFrame(data)
    
    if df.empty:
        print(f'No data available for {metric} to plot.')
        return
    
    title = f'Boxplot of {metric.capitalize()} by {hue_val.capitalize()} and {x_label.capitalize()}'
    filename = f'{metric}_boxplot.png'

    # Creating the boxplot using seaborn for easy and attractive visualization
    plt.figure(figsize=figsize, dpi=dpi)
    sns.boxplot(x=x_label, y='Value', hue=hue_val, data=df)
    plt.title(f'Boxplot of {metric} by Exit and Model')
    plt.xlabel(x_label)
    plt.ylabel(metric.capitalize())
    plt.grid(grid)
    if legend:
        plt.legend(title=hue_val)
    else:
        plt.legend().remove()
        
    if save_file:
        # Save the plot to the specified directory
        plot_file = os.path.join(out_dir, filename)
        plt.savefig(plot_file)
        print(f'\033[3m{title}\033[0m saved to {plot_file}')
    else:
        plt.show()
    plt.close()


def barplot_metrics(metric: str, models_metrics: dict, out_dir: str,
                    x_label: str = 'Model', hue_val: str = "Exit",
                    grid: bool = True, legend: bool = False,
                    figsize: tuple = (12, 6), dpi: int = 100,
                    save_file: bool = True) -> None:
    """
    Creates and saves a bar plot for the specified metric across different models, categorized by a hue variable
    
    Parameters:
    - metric (str): The metric to plot
    - models_metrics (dict): Nested dictionary with model names as keys at the top level, and 'phases' dict which further maps to 'exits' dict containing metric values
    - out_dir (str): Directory where the bar plot image will be saved
    - x_label (str): Label for the x-axis. Default is 'Model'
    - hue_val (str): The column name to be used as the hue variable in the plot. Default is 'Exit'
    - grid (bool): Whether to display a grid on the plot. Default is True
    - legend (bool): Whether to display a legend on the plot. Default is False
    - figsize (tuple): Figure size in inches as a tuple (width, height). Default is (12, 6)
    - dpi (int): Dots per inch (resolution) of the figure. Default is 100
    - save_file (bool): If True, saves the plot to a file; if False, displays the plot. Default is True

    Returns:
    - None
    """

    # Collecting data for DataFrame
    data = []
    phase = 'test'

    for model_name in models_metrics:
        for exit in models_metrics[model_name][phase]:
            metric_values = models_metrics[model_name][phase][exit].get(metric)
            if metric_values is not None:
                data.append({
                    x_label: model_name,
                    hue_val: f'Exit {exit}',  # Labeling the exit
                    'Value': metric_values
                })

    
    df = pd.DataFrame(data)
    
    title = f'Barplot of {metric.capitalize()} by {hue_val.capitalize()} and {x_label.capitalize()}'
    filename = f'{metric}_barplot.png'

    if df.empty:
        print(f'No data available for {metric} to plot.')
        return

    # Creating the boxplot using seaborn for easy and attractive visualization
    plt.figure(figsize=figsize, dpi=dpi)

    sns.barplot(x=x_label, y='Value', hue=hue_val, data=df)
    plt.title(f'Barplot of {metric} by Exit and Model')
    plt.xlabel(x_label.capitalize())
    plt.ylabel(metric.capitalize())
    plt.grid(grid)
    if legend:
        plt.legend(title=hue_val)
    else:
        plt.legend().remove()
        
    # Save the plot to the specified directory
    if save_file:
        plot_file = os.path.join(out_dir, filename)
        plt.savefig(plot_file)
        print(f'\033[3m{title}\033[0m saved to {plot_file}')
    else:
        plt.show()
    plt.close()


def plot_cumulative_count(data_dict, min_threshold=None, max_threshold=None,
                          x_label='Threshold', y_label='Number of Samples',
                          title='Progressive Cumulative Count Above Each Threshold',
                          fig_size:tuple=(10, 6), dpi:int=100,
                          grid=True, legend:bool=True, save_file:bool=True, filename='cumulative_count.png'):
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
    - fig_size: size of the figure in inches.
    - dpi: resolution of the figure in dots per inch.
    - grid: whether to display grid lines on the plot.
    - legend: whether to display a legend on the plot.
    - save_file: if True, saves the plot to a file; otherwise, displays the plot.
    - filename: name of the file to save the plot to.

    Returns:
    - None
    """

    if min_threshold is None or max_threshold is None:
        all_data = np.concatenate(list(data_dict.values()))
        if min_threshold is None:
            min_threshold = np.min(all_data)
        if max_threshold is None:
            max_threshold = np.max(all_data)
    
    plt.figure(figsize=fig_size, dpi=dpi)
    # Plot data for each model using the global min and max threshold
    for model_name, data in data_dict.items():
        thresholds = np.linspace(min_threshold, max_threshold, 100)
        counts = [np.sum(data > threshold) for threshold in thresholds]
        
        plt.plot(thresholds, counts, label=model_name)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(grid)
    if legend:
        plt.legend()
    else:
        plt.legend().remove()

    # Save the plot or display it
    if save_file:
        plt.savefig(filename)
        print(f'\033[3m{title}\033[0m saved to {filename}')
    else:
        plt.show()

    plt.close()


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


def plot_dictionary_subplots(data_dict: dict, super_title: str = 'Subplots for Dictionary',
                             x_label: str = 'X-axis', y_label: str = 'Y-axis',
                             title_prefix: str = 'Plot for Key', grid: bool = False,
                             figsize: tuple = (10, 5), dpi: int = 100, marker_size: int = 10,
                             save_file: bool = True, filename: str = "difference_plot.png",
                             out_dir: str = "") -> None:
    """
    Creates a figure with a subplot for each key-value pair in the input dictionary. Each subplot plots the data from the list of tuples, where the first element of each tuple is y, and the second is x.

    Parameters:
    - data_dict (dict): Dictionary with keys as numbers and values as lists of tuples.
    - super_title (str): Title for the entire figure. Default is 'Subplots for Dictionary'
    - x_label (str): Label for the x-axis. Default is 'X-axis'
    - y_label (str): Label for the y-axis. Default is 'Y-axis'
    - title_prefix (str): Prefix for the title of each subplot. Default is 'Plot for Key'
    - grid (bool): Whether to display a grid on each subplot. Default is False
    - figsize (tuple): Overall figure size in inches as a tuple (width, height). Default is (10, 5)
    - dpi (int): Dots per inch (resolution) of the figure. Default is 100
    - save_file (bool): If True, saves the plot to a file; if False, displays the plot. Default is True
    - filename (str): Name of the file to save the plot. Required if save_file is True
    - out_dir (str): Directory where the plot image will be saved if save_file is True. Required if save_file is True

    Returns:
    - None
    """

    num_plots = len(data_dict)
    if num_plots == 0:
        print("The input dictionary is empty. No plots to display.")
        return

    # Determine the number of rows and columns for subplots
    ncols = min(2, num_plots)  # Up to 2 plots per row
    nrows = (num_plots + ncols - 1) // ncols  # Compute number of rows needed

    # adjust the figure size based on the number of subplots
    figsize = (figsize[0], figsize[1] * nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, dpi=dpi)
    axes = axes.flatten() if num_plots > 1 else [axes]  # Ensure axes is iterable

    for idx, (key, value_list) in enumerate(data_dict.items()):
        ax = axes[idx]
        y_values = [t[0] for t in value_list]
        x_values = [t[1] for t in value_list]
        ax.scatter(x_values, y_values, marker='o', s=marker_size)  # Use scatter plot to plot points without connecting lines
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{title_prefix} {key}')
        ax.grid(grid)

    # Hide any unused subplots
    for idx in range(len(data_dict), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.suptitle(super_title, y=1.02)  # Adjust the y position for the super title

    if save_file:
        if not filename:
            raise ValueError("Filename must be provided when save_file is True.")
        if not out_dir:
            raise ValueError("Output directory 'out_dir' must be specified when 'save_file' is True.")
        plot_file = os.path.join(out_dir, filename)
        plt.savefig(plot_file)
        print(f'Plot saved to {plot_file}')
    else:
        plt.show()
    plt.close()

def plot_metric_vs_confidence(data_dict: dict, metric: str,
                              min_confidence: float = 0.2,
                              x_label: str = 'Confidence',
                              title: str = "Metric vs Confidence",
                              grid: bool = True,
                              figsize: tuple = (12, 6),
                              dpi: int = 100,
                              save_file: bool = True,
                              filename: str = "metric_vs_confidence.png",
                              out_dir: str = "") -> None:
    """
    Creates a single plot for all models. The x-axis represents confidence levels,
    and the y-axis shows the average of the specified metric for all data points with at least that confidence value.
    A minimum confidence can be specified to filter the data.

    Parameters:
    - data_dict (dict): Dictionary with model names as keys and values as dictionaries.
                        Each value dictionary contains the metric values as a list of tuples.
                        Each tuple contains (metric_value, confidence_value, weight).
    - metric (str): The metric to plot.
    - min_confidence (float): The minimum confidence level to start plotting from. Default is 0.5.
    - x_label (str): Label for the x-axis. Default is 'Confidence'.
    - title (str): Title of the plot. Default is 'Metric vs Confidence'.
    - grid (bool): Whether to display grid lines on the plot. Default is True.
    - figsize (tuple): Figure size in inches as a tuple (width, height). Default is (12, 6).
    - dpi (int): Dots per inch (resolution) of the figure. Default is 100.
    - save_file (bool): If True, saves the plot to a file; if False, displays the plot. Default is True.
    - filename (str): Name of the file to save the plot. Default is 'metric_vs_confidence.png'.
    - out_dir (str): Directory where the plot image will be saved if save_file is True. Required if save_file is True.

    Returns:
    - None
    """

    plt.figure(figsize=figsize, dpi=dpi)

    # Collect all confidence values across models
    all_confidences = []

    for model_name, metrics_dict in data_dict.items():
        if metric not in metrics_dict:
            print(f"Metric '{metric}' not found for model '{model_name}'.")
            continue

        metric_tuples = metrics_dict[metric]
        confidences = [t[1] for t in metric_tuples]
        all_confidences.extend(confidences)

    all_confidences = np.array(all_confidences)
    if min_confidence is not None:
        # Filter out confidence values below the minimum confidence
        all_confidences = all_confidences[all_confidences >= min_confidence]

    # Get sorted unique confidence levels
    common_confidences = np.unique(all_confidences)
    common_confidences.sort()

    if len(common_confidences) == 0:
        print("No common confidence levels found above the specified minimum.")
        return

    for model_name, metrics_dict in data_dict.items():
        if metric not in metrics_dict:
            continue

        metric_tuples = metrics_dict[metric]
        metric_values = np.array([t[0] for t in metric_tuples])
        confidence_values = np.array([t[1] for t in metric_tuples])
        weights = np.array([t[2] for t in metric_tuples])

        if min_confidence is not None:
            # Filter data based on minimum confidence
            mask = confidence_values >= min_confidence
            metric_values = metric_values[mask]
            confidence_values = confidence_values[mask]
            weights = weights[mask]

        if len(confidence_values) == 0:
            print(f"No data for model '{model_name}' above the minimum confidence.")
            continue

        avg_metric_values = []
        confidence_levels = []

        for conf in common_confidences:
            mask = confidence_values >= conf
            if np.any(mask):
                # Compute weighted average of metric values
                weighted_avg = np.average(metric_values[mask], weights=weights[mask])
                avg_metric_values.append(weighted_avg)
                confidence_levels.append(conf)

        # Plot data for the current model without markers
        plt.plot(confidence_levels, avg_metric_values, label=model_name)

    plt.xlabel(x_label)
    plt.ylabel(f'Average of {metric}')
    if title is None:
        plot_title = f'Average {metric} vs Confidence'
    else:
        plot_title = title
    plt.title(plot_title)
    plt.grid(grid)
    plt.legend()

    if save_file:
        if not filename:
            filename = f'{metric}_vs_confidence.png'
        if not out_dir:
            raise ValueError("Please specify 'out_dir' when 'save_file' is True.")
        plot_file = os.path.join(out_dir, filename)
        plt.savefig(plot_file)
        print(f'Plot saved to {plot_file}')
    else:
        plt.show()
    plt.close()