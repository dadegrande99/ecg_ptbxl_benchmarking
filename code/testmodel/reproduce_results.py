from models import MODEL_LIST
import argparse
import os
import json
import pandas as pd
import numpy as np
from exceptions import DataDirectoryError
from utils.loader import get_dataloader, import_ptbxl, split_data
from utils import select_device
from utils.metrics import compute_loss, compute_metrics
import torch
import torch.nn.functional as F
import torch.optim as optim


def train(model, device, train_loader, optimizer, epoch, output_dir, model_name, threshold=0.5):
    model.train()
    all_targets = []
    all_preds = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = compute_loss(output, target)
        loss.backward()
        optimizer.step()
        
        all_targets.extend(target.cpu().numpy())
        all_preds.extend(output.detach().cpu().numpy())

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)] Loss: {loss.item():.6f}', end='\r')

    predictions = pd.DataFrame({
        "target" : all_targets,
        "prediction" : all_preds
    })

    predictions.to_csv(os.path.join(output_dir, 'models', model_name, 'train', f'{str(epoch).zfill(3)}.csv'), index=False)
    
    avg_auc, _, _, _, _, _ = compute_metrics(torch.tensor(np.array(all_preds)), torch.tensor(np.array(all_targets)), threshold=threshold)
    print(f'Train Epoch: {epoch} Loss: {loss.item():.4f}, AUC (avg): {avg_auc:.2f}')


def validate(model, device, val_loader, epoch, output_dir, model_name, thresholds=np.arange(0.1, 1, 0.1)):
    model.eval()
    val_loss = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = compute_loss(output, target)
            val_loss += loss.item()
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(output.cpu().numpy())

    val_loss /= len(val_loader)

    best_auc = 0
    best_results = {}
    best_threshold = 0
    for threshold in thresholds:
        avg_auc, _, _, _, _, _ = compute_metrics(torch.tensor(np.array(all_preds)), torch.tensor(np.array(all_targets)), threshold=threshold)
        if avg_auc > best_auc:
            best_auc = avg_auc
            best_threshold = threshold
    
    predictions = pd.DataFrame({
        "target" : all_targets,
        "prediction" : all_preds
    })
    predictions.to_csv(os.path.join(output_dir, 'models', model_name, 'val', f'{str(epoch).zfill(3)}.csv'), index=False)
    
    print(f'Validation set: Average loss: {val_loss:.4f}, AUC (avg): {best_auc:.2f}')

    return best_auc, best_threshold, model.state_dict()


def test(model, device, test_loader, output_dir, model_name, threshold=0.5):
    model.eval()
    test_loss = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = compute_loss(output, target)
            test_loss += loss.item()
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(output.cpu().numpy())

    test_loss /= len(test_loader)
    avg_auc, avg_f1, avg_acc, aucs, f1_scores, accuracies  = compute_metrics(torch.tensor(np.array(all_preds)), torch.tensor(np.array(all_targets)), threshold=threshold)

    predictions = pd.DataFrame({
        "target" : all_targets,
        "prediction" : all_preds
    })
    predictions.to_csv(os.path.join(output_dir, 'models', model_name, 'test.csv'), index=False)
    
    print(f'- Average loss: {test_loss:.4f}')
    print(f'- AUC (avg): {avg_auc:.4f}, AUC class 0: {aucs[0]:.4f}, AUC class 1: {aucs[1]:.4f}')
    print(f'- F1 (avg): {avg_f1:.4f}, F1 class 0: {f1_scores[0]:.4f}, F1 class 1: {f1_scores[1]:.4f}')
    print(f'- Accuracy (avg): {avg_acc:.4f}, Accuracy class 0: {accuracies[0]:.4f}, Accuracy class 1: {accuracies[1]:.4f}')

    return avg_auc, avg_f1, avg_acc, aucs, f1_scores, accuracies


def main():

    parser = argparse.ArgumentParser(description='Train and test models.')
    parser.add_argument('-d', '--data-dir', type=str, default='../../data/ptbxl/',
                        help='Directory containing the ptbxl data')
    parser.add_argument('-o', '--output-dir', type=str,
                        default='../../output/', help='Directory to save the results')
    parser.add_argument('-c', '--config', type=str,
                        default='config.json', help='Path to the configuration file')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise DataDirectoryError(data_dir)

    config_path = args.config
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}

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
    _, ptbxl = import_ptbxl(path=data_dir, clean=clean)

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
                                batch_size=batch_size,  in_channels=in_channels)
    test_loader = get_dataloader(test_df['raw_data'].tolist(), test_df[['MI', 'NORM']].values,
                                 batch_size=batch_size, in_channels=in_channels, shuffle=False)

    device = select_device()

    thresholds = np.arange(0.1, 1, 0.1)

    for model_name, ModelClass in MODEL_LIST:
        print(f"\nTraining and testing model: {model_name}")
        model = ModelClass(in_channels=in_channels, num_classes=num_classes, dropout_rate=dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Create directories to save the results
        if not os.path.isdir(f'{output_dir}/models/{model_name}'):
            os.makedirs(f'{output_dir}/models/{model_name}')
        for el in ['train', 'val']:
            if not os.path.isdir(f'{output_dir}/models/{model_name}/{el}'):
                os.makedirs(f'{output_dir}/models/{model_name}/{el}')

        best_auc = 0
        best_epoch = 0
        best_threshold = thresholds[0]
        best_model_state = None
        for epoch in range(1, num_epochs + 1):
            train(model, device, train_loader, optimizer, epoch, output_dir, model_name, threshold=best_threshold)
            val_auc, threshold, model_state = validate(model, device, val_loader, epoch, output_dir, model_name, thresholds=thresholds)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                best_threshold = threshold
                print(best_threshold, type(best_threshold))
                best_model_state = model_state
        print(f"Best model with AUC: {best_auc:.4f} at epoch {best_epoch}")

        # Save the best model state at the end of all epochs
        if best_model_state is not None:
            torch.save(best_model_state, os.path.join(output_dir, "models", model_name, "best_model.pth"))
            print(f"Saved best model with AUC: {best_auc:.4f}")

        print(f"Final evaluation on the test set for model: {model_name}")
        test_auc = test(model, device, test_loader, output_dir, model_name, threshold=best_threshold)

        # Save the best threshold, test AUC and best epoch in a JSON file
        results = {
            "best_threshold": best_threshold,
            "test_auc": test_auc,
            "best_epoch": best_epoch
        }
        with open(os.path.join(output_dir, "models", model_name, "results.json"), 'w') as f:
            json.dump(results, f)



if __name__ == '__main__':
    main()
