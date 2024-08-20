from models import MODEL_LIST
import argparse
import os
import json
from exceptions import DataDirectoryError
from utils.loader import get_dataloader, import_ptbxl, split_data
from results_analysis import main as analyze_results
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import torch
import lightning as L

def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Train and test models.')
    parser.add_argument('-d', '--data-dir', type=str, default='../../data/ptbxl/',
                        help='Directory containing the ptbxl data')
    parser.add_argument('-o', '--output-dir', type=str,
                        default='../../output/', help='Directory to save the results')
    parser.add_argument('-c', '--config', type=str,
                        default='config.json', help='Path to the configuration file')
    args = parser.parse_args()

    # Check if the data directory exists
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise DataDirectoryError(data_dir)

    # Load the configuration file
    config_path = args.config
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    # Get the configuration parameters
    batch_size = config.get('batch_size', 128)
    learning_rate = config.get('learning_rate', 0.001)
    clean = config.get('clean', False)
    num_epochs = config.get('num_epochs', 50)
    in_channels = config.get('in_channels', 12)
    num_classes = config.get('num_classes', 2)
    train_fold = config.get('train_fold', 8)
    test_fold = config.get('test_fold', 9)
    val_fold = config.get('val_fold', 10)
    dropout_rate = config.get('dropout_rate', 0.05)

    config = {
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'clean': clean,
        'num_epochs': num_epochs,
        'in_channels': in_channels,
        'num_classes': num_classes,
        'train_fold': train_fold,
        'test_fold': test_fold,
        'val_fold': val_fold,
        'dropout_rate': dropout_rate
    }

    # import data
    print("Importing data ...")
    _, ptbxl = import_ptbxl(path=data_dir, clean=clean)

    # Create the output directory
    print("\nCreating output directory if not exist...")
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' created.")
    if not os.path.isdir(f'{output_dir}/models'):
        os.makedirs(f'{output_dir}/models')
        print(f"Model directory '{os.path.join(output_dir, "models")}' created.")
    if not os.path.isdir(f'{output_dir}/data'):
        os.makedirs(f'{output_dir}/data')
        print(f"Data directory '{os.path.join(output_dir, "data")}' created.")
    print()

    # Save the configuration file in the output directory
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # split data
    train_df, val_df, test_df = split_data(
        ptbxl, folds=[train_fold, val_fold, test_fold])

    # Save the split data
    train_df[['MI']].to_csv(os.path.join(
        output_dir, 'data', 'y_training.csv'), index=False)
    val_df[['MI']].to_csv(os.path.join(
        output_dir, 'data', 'y_validation.csv'), index=False)
    test_df[['MI']].to_csv(os.path.join(
        output_dir, 'data', 'y_test.csv'), index=False)

    # create DataLoaders
    train_loader = get_dataloader( train_df['raw_data'].tolist(), train_df[['MI', 'NORM']].values, 
                                  batch_size=batch_size, in_channels=in_channels)
    val_loader = get_dataloader(val_df['raw_data'].tolist(), val_df[['MI', 'NORM']].values,
                                batch_size=batch_size,  in_channels=in_channels, shuffle=False)
    test_loader = get_dataloader(test_df['raw_data'].tolist(), test_df[['MI', 'NORM']].values,
                                 batch_size=batch_size, in_channels=in_channels, shuffle=False)

    for model_name, ModelClass in MODEL_LIST:
        early_stop_callback = EarlyStopping(monitor="val_avg_auc", min_delta=0.00, patience=3, verbose=False, mode="max")
        print(f"\n\n\nPrepare settings for {model_name} ...")
        model = ModelClass(in_channels=in_channels, num_classes=num_classes, dropout_rate=dropout_rate, learning_rate=learning_rate)
        trainer = L.Trainer(limit_train_batches=batch_size, max_epochs=num_epochs, callbacks=[early_stop_callback], devices=1, default_root_dir=os.path.abspath(f'{output_dir}/models/{model_name}'))

        print(f"\n\nTraining model {model_name} ...\n")

        with torch.no_grad():
            trainer.fit(model, train_loader, val_loader)
            trainer.test(model, test_loader)
            trainer.save_checkpoint(f'{output_dir}/models/{model_name}/checkpoint.ckpt')
        print()

    print("Done!")
    # print("\n\n\nAnalyzing results ...\n")
    # analyze_results(output_dir)
        

if __name__ == '__main__':
    main()