from models import MODEL_LIST
import argparse
import os
import json
import pandas as pd
from exceptions import DataDirectoryError
from utils.loader import get_dataloader, import_ptbxl, split_data
from utils import select_device
import torch
import torch.nn.functional as F
import torch.optim as optim

def train(model, device, train_loader, optimizer, epoch, output_dir, model_name):
    model.train()
    all_targets = []
    all_preds = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        pred = output.argmax(dim=1, keepdim=True)
        all_targets.extend(target.cpu().numpy())
        all_preds.extend(pred.cpu().numpy())

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}', end='\r')


    predictions = pd.DataFrame({
        "target": all_targets,
        "prediction": all_preds
    })
    predictions.to_csv(os.path.join(output_dir, model_name, 'train', f'{str(epoch).zfill(3)}.csv'), index=False)
    
    print(f'Train Epoch: {epoch} Loss: {loss.item():.4f}, Accuracy: {100. * sum(all_preds == all_targets) / len(all_targets):.2f}%')# type: ignore


def validate(model, device, val_loader, epoch, output_dir, model_name):
    model.eval()
    val_loss = 0
    correct = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    predictions = pd.DataFrame({
        "target": all_targets,
        "prediction": all_preds
    })
    predictions.to_csv(os.path.join(output_dir, model_name, 'val', f'{str(epoch).zfill(3)}.csv'), index=False)
    
    print(f'\nValidation set: Average loss: {val_loss:.4f}, Accuracy: {100. * correct / len(val_loader.dataset):.2f}%\n')


def test(model, device, test_loader, output_dir, model_name):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())

    test_loss /= len(test_loader.dataset)
    predictions = pd.DataFrame({
        "target": all_targets,
        "prediction": all_preds
    })
    predictions.to_csv(os.path.join(output_dir, model_name, 'test.csv'), index=False)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {100. * correct / len(test_loader.dataset):.2f}%\n')


def main():

    parser = argparse.ArgumentParser(description='Train and test models.')
    parser.add_argument('-d', '--data-dir', type=str, default='../../data/ptbxl/', help='Directory containing the ptbxl data')
    parser.add_argument('-o', '--output-dir', type=str, default='../../output/', help='Directory to save the results')
    parser.add_argument('-c', '--config', type=str, default='config.json', help='Path to the configuration file')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        raise DataDirectoryError(data_dir)
    raw, ptbxl = import_ptbxl(path=data_dir)

    config_path = args.config
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    batch_size = config.get('batch_size', 128)
    learning_rate = config.get('learning_rate', 0.001)
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
        'num_epochs': num_epochs,
        'in_channels': in_channels,
        'num_classes': num_classes,
        'train_fold': train_fold,
        'test_fold': test_fold,
        'val_fold': val_fold,
        'dropout_rate': dropout_rate
    }

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' created.")
    if not os.path.isdir(f'{output_dir}/models'):
        os.makedirs(f'{output_dir}/models')
        print(f"Model directory '{output_dir}/models' created.")
    if not os.path.isdir(f'{output_dir}/data'):
        os.makedirs(f'{output_dir}/data')
        print(f"Data directory '{output_dir}/data' created.")

    # Split data in train, val e test
    train_df, val_df, test_df = split_data(ptbxl, folds=[train_fold, val_fold, test_fold])

    # create DataLoaders
    train_loader = get_dataloader(train_df['raw_data'].tolist(), train_df['MI'].values, batch_size=batch_size, in_channels=in_channels)
    val_loader = get_dataloader(val_df['raw_data'].tolist(), val_df['MI'].values, batch_size=batch_size, in_channels=in_channels)
    test_loader = get_dataloader(test_df['raw_data'].tolist(), test_df['MI'].values, batch_size=batch_size, in_channels=in_channels, shuffle=False)

    device = select_device()

    for model_name, ModelClass in MODEL_LIST:
        print(f"Training and testing model: {model_name}")
        model = ModelClass(in_channels=in_channels, num_classes=num_classes, dropout_rate=dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Create directories to save the results
        if not os.path.isdir(f'{output_dir}/models/{model_name}'):
            os.makedirs(f'{output_dir}/models{model_name}')
        for el in ['train', 'val']:
            if not os.path.isdir(f'{output_dir}/models/{model_name}/{el}'):
                os.makedirs(f'{output_dir}/models/{model_name}/{el}')


        for epoch in range(1, num_epochs + 1):
            train(model, device, train_loader, optimizer, epoch, output_dir, model_name)
            validate(model, device, val_loader, epoch, output_dir, model_name)
        
        print(f"Final evaluation on the test set for model: {model_name}")
        test(model, device, test_loader, output_dir, model_name)

if __name__ == '__main__':
    main()


