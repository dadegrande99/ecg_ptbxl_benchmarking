from abc import ABC, abstractmethod
import json
import torch
from torch import optim, nn
import lightning as L
from utils import compute_loss, compute_metrics
import numpy as np
import pandas as pd
import os

class BaseModel(ABC, L.LightningModule):
    def __init__(self, 
                 in_channels: int, 
                 num_classes: int, 
                 dropout_rate: float = 0.5, 
                 learning_rate: float = 0.1, 
                 thresholds: np.ndarray = np.arange(0.1, 1, 0.1, dtype=np.float32)):
        super(BaseModel, self).__init__()
        self.loss = compute_loss
        self.learning_rate = learning_rate
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []
        self.thresholds = thresholds
        self.best_auc = 0.0
        self.best_threshold = 0.5
        self.entropy_threshold = 0.5

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def forward_training(self, batch, batch_idx):
        pass

    def should_exit(self, predictions, threshold):
        predictions_np = predictions.detach().cpu().numpy()
        entropy = custom_entropy_formula(predictions_np)
        return entropy < threshold


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        losses = [self.loss(output, y) for output in y_hat]
        total_loss = sum(losses) / len(losses)
        
        self.log('train_loss', total_loss)
        self.training_outputs.append((y_hat, y))
        return total_loss
    
    def on_train_epoch_end(self) -> None:
        y_hats, ys = zip(*self.training_outputs)
        ys = torch.cat(ys, dim=0).cpu()
        
        for i in range(len(y_hats[0])):
            outputs = [y_hat[i].detach().cpu() for y_hat in y_hats]
            outputs = torch.cat(outputs, dim=0)
            
            # Create a folder for each exit
            exit_path = os.path.join(self.trainer.default_root_dir, f'train/exit_{i}')
            os.makedirs(exit_path, exist_ok=True)
            
            # Save the outputs and targets
            outputs_dict = {f"output_{j}": outputs[:, j].numpy() for j in range(outputs.shape[1])}
            targets_dict = {f"target_{j}": ys[:, j].numpy() for j in range(ys.shape[1])}
            
            train_df = pd.DataFrame({
                **outputs_dict,
                **targets_dict
            })
            train_df.to_csv(f'{exit_path}/{self.current_epoch:03d}.csv', index=False)
            
            # Compute and log metrics
            avg_auc, avg_f1, avg_acc, aucs, f1_scores, accuracies = compute_metrics(outputs, ys, self.best_threshold)
            self.log(f'train_avg_auc_{i}', avg_auc)
            self.log(f'train_avg_f1_{i}', avg_f1)
            self.log(f'train_avg_acc_{i}', avg_acc)
        
        self.training_outputs.clear()
    

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.loss(output, target)
        self.log("val_loss", loss)
        self.validation_outputs.append((output.detach().cpu(), target.detach().cpu()))
        return loss

    def on_validation_epoch_end(self) -> None:
        # Aggregate all validation outputs
        outputs, targets = zip(*self.validation_outputs)
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        
        # Compute epoch-level metrics
        best_epoch = {
            "avg_auc" : 0.0,
            "avg_f1" : 0.0,
            "avg_acc" : 0.0
        }
        for threshold in self.thresholds:
            avg_auc, avg_f1, avg_acc, aucs, f1_scores, accuracies = compute_metrics(outputs, targets, threshold) # type: ignore
            if avg_auc > best_epoch["avg_auc"]:
                best_epoch["avg_auc"] = avg_auc
                best_epoch["avg_f1"] = avg_f1
                best_epoch["avg_acc"] = avg_acc
                if avg_auc > self.best_auc:
                    self.best_auc = avg_auc
                    self.best_threshold = threshold

        
        # Log aggregated metrics
        self.log('val_avg_auc', best_epoch["avg_auc"])
        self.log('val_avg_f1', best_epoch["avg_f1"])
        self.log('val_avg_acc', best_epoch["avg_acc"])

        # Create dictionaries for output and target columns
        outputs_dict = {f"output_{i}": outputs[:, i].numpy() for i in range(outputs.shape[1])}
        targets_dict = {f"target_{i}": targets[:, i].numpy() for i in range(targets.shape[1])}
        
        # Combine dictionaries into a DataFrame
        val_df = pd.DataFrame({
            **outputs_dict,
            **targets_dict
        })

        # Save the DataFrame to a CSV file
        full_path = os.path.join(self.trainer.default_root_dir, 'val')
        os.makedirs(full_path, exist_ok=True)
        val_df.to_csv(f'{full_path}/{self.current_epoch:03d}.csv', index=False)
        
        # Clear the list for the next epoch
        self.validation_outputs.clear()

    
    def test_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = self.loss(output, target)
        self.log("test_loss", loss)
        self.test_outputs.append((output.detach().cpu(), target.detach().cpu()))
        return loss
    
    def on_test_epoch_end(self) -> None:
        # Aggregate all test outputs
        outputs, targets = zip(*self.test_outputs)
        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        
        # Compute epoch-level metrics
        avg_auc, avg_f1, avg_acc, aucs, f1_scores, accuracies = compute_metrics(outputs, targets, self.best_threshold) # type: ignore
        
        # Log aggregated metrics
        self.log('test_avg_auc', avg_auc) # type: ignore
        self.log('test_avg_f1', avg_f1) # type: ignore
        self.log('test_avg_acc', avg_acc) # type: ignore

        
        # Create dictionaries for output and target columns
        outputs_dict = {f"output_{i}": outputs[:, i].numpy() for i in range(outputs.shape[1])}
        targets_dict = {f"target_{i}": targets[:, i].numpy() for i in range(targets.shape[1])}
        
        # Combine dictionaries into a DataFrame
        test_df = pd.DataFrame({
            **outputs_dict,
            **targets_dict
        })

        # Save the DataFrame to a CSV file
        full_path = os.path.join(self.trainer.default_root_dir)
        os.makedirs(full_path, exist_ok=True)
        test_df.to_csv(f'{full_path}/test.csv', index=False)

        # Save model values
        values = {
            "best_auc": float(self.best_auc),
            "best_threshold": float(self.best_threshold)
        }
        with open(f'{full_path}/values.json', 'w') as f:
            json.dump(values, f, indent=4)

        # Clear the list for the next epoch
        self.test_outputs.clear()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def predict_step(self, batch):
        inputs, target = batch
        return self.model(inputs, target)
    
def custom_entropy_formula(predictions: np.array) -> np.array: # type: ignore
    predictions = np.clip(predictions, 1e-9, 1 - 1e-9)  # Ensure values are in the range [1e-9, 1-1e-9]
    predictions = np.mean(predictions, axis=0)  # Average predictions across batch
    entropy = -np.nansum(predictions * np.log(predictions), axis=0) / np.log(predictions.shape[0])
    return entropy
