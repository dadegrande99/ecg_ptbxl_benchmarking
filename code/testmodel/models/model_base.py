from abc import ABC, abstractmethod
import json
import torch
from torch import optim, nn
import lightning as L
from utils.metrics import compute_loss_ee, compute_metrics, custom_entropy_formula
import numpy as np
import pandas as pd
import os

class BaseModelEE(ABC, L.LightningModule):
    def __init__(self, 
                 in_channels: int, 
                 num_classes: int, 
                 dropout_rate: float = 0.5, 
                 learning_rate: float = 0.1, 
                 thresholds: np.ndarray = np.arange(0.1, 1, 0.1, dtype=np.float32)):
        super(BaseModelEE, self).__init__()
        self.loss = compute_loss_ee
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
        self.modules_EE = nn.ModuleList()
        self.exits = nn.ModuleList()
        self.weights_ee = []
        self.test_all_exits = True
        self.exits_used = []

    def forward(self, x):
        x = self.forward_intro(x)

        x, outputs = self.forward_modules(x)

        if len(outputs) <= len(self.modules_EE):
            outputs.append(self.forward_final(x))

        return tuple(outputs)

    @abstractmethod
    def forward_intro(self, x):
        # it will return the data to be used in the forward
        pass

    def forward_modules(self, x):
        outputs = []
        exit_used = -1

        for i, layer in enumerate(self.modules_EE):
            x = layer(x)
            out = x.mean(dim=2)
            outputs.append(torch.sigmoid(self.exits[i](out)))
            should_exit = self.should_exit(outputs[-1], self.entropy_threshold)
            if should_exit and exit_used == -1:
                exit_used = i
            if not(self.training) and should_exit:
                for _ in range(i, len(self.exits)):
                    outputs.append(outputs[-1])
                self.exits_used[i] += 1
                return x, outputs
        self.exits_used[exit_used] += 1
        return x, outputs

    @abstractmethod
    def forward_final(self, x):
        pass


    def should_exit(self, predictions, threshold):
        predictions_np = predictions.detach().cpu().numpy()
        entropy = custom_entropy_formula(predictions_np)
        return entropy < threshold


    def training_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self(inputs)
        total_loss = self.loss(outputs, target, self.weights_ee)
        self.log('train_loss', total_loss)
        self.training_outputs.append((outputs, target))
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
            avg_auc, avg_f1, avg_acc, avg_recall, aucs, f1_scores, accuracies, recall = compute_metrics(outputs, ys, self.best_threshold)
            self.log(f'train_avg_auc_{i}', avg_auc)
            self.log(f'train_avg_f1_{i}', avg_f1)
            self.log(f'train_avg_acc_{i}', avg_acc)
        
        self.training_outputs.clear()
    

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self(inputs)
        total_loss = self.loss(outputs, target, self.weights_ee)
        self.log("val_loss", total_loss)
        self.validation_outputs.append((outputs, target))
        return total_loss

    def on_validation_epoch_end(self) -> None:
        y_hats, targets = zip(*self.validation_outputs)
        targets = torch.cat(targets, dim=0).cpu()

        best_epoch = {
            "avg_auc": 0.0,
            "avg_f1": 0.0,
            "avg_acc": 0.0
        }
        
        for i in range(len(y_hats[0])):
            outputs = [y_hat[i].detach().cpu() for y_hat in y_hats]
            outputs = torch.cat(outputs, dim=0)
            
            # Create a folder for each exit
            exit_path = os.path.join(self.trainer.default_root_dir, f'val/exit_{i}')
            os.makedirs(exit_path, exist_ok=True)
            
            # Save the outputs and targets
            outputs_dict = {f"output_{j}": outputs[:, j].numpy() for j in range(outputs.shape[1])}
            targets_dict = {f"target_{j}": targets[:, j].numpy() for j in range(targets.shape[1])}
            
            val_df = pd.DataFrame({
                **outputs_dict,
                **targets_dict
            })
            val_df.to_csv(f'{exit_path}/{self.current_epoch:03d}.csv', index=False)
            
            # Compute and log metrics
            best_exit = {
                "avg_auc": 0.0,
                "avg_f1": 0.0,
                "avg_acc": 0.0
            }
            for threshold in self.thresholds:
                avg_auc, avg_f1, avg_acc, avg_recall, aucs, f1_scores, accuracies, recall = compute_metrics(outputs, targets, threshold) # type: ignore
                if avg_auc > best_exit["avg_auc"]:
                    best_exit["avg_auc"] = avg_auc
                    best_exit["avg_f1"] = avg_f1
                    best_exit["avg_acc"] = avg_acc
                    if avg_auc > best_epoch["avg_auc"]:
                        best_epoch["avg_auc"] = avg_auc
                        best_epoch["avg_f1"] = avg_f1
                        best_epoch["avg_acc"] = avg_acc
                        if avg_auc > self.best_auc:
                            self.best_auc = avg_auc
                            self.best_threshold = threshold
            
            self.log(f'val_avg_auc_{i}', best_exit["avg_auc"])
            self.log(f'val_avg_f1_{i}', best_exit["avg_f1"])
            self.log(f'val_avg_acc_{i}', best_exit["avg_acc"])

        self.log(f'val_avg_auc', best_epoch["avg_auc"])
        self.log(f'val_avg_f1', best_epoch["avg_f1"])
        self.log(f'val_avg_acc', best_epoch["avg_acc"])
        
        self.validation_outputs.clear()

    
    def test_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self(inputs)
        total_loss = self.loss(outputs, target, self.weights_ee)
        self.log("test_loss", total_loss)
        self.test_outputs.append((outputs, target))
        return total_loss
    
    def on_test_epoch_end(self) -> None:
        y_hats, ys = zip(*self.test_outputs)
        ys = torch.cat(ys, dim=0).cpu()
        
        for i in range(len(y_hats[0])):
            outputs = [y_hat[i].detach().cpu() for y_hat in y_hats]
            outputs = torch.cat(outputs, dim=0)
            
            # Create a folder for each exit
            test_path = os.path.join(self.trainer.default_root_dir, 'test')
            os.makedirs(test_path, exist_ok=True)
            
            # Save the outputs and targets
            outputs_dict = {f"output_{j}": outputs[:, j].numpy() for j in range(outputs.shape[1])}
            targets_dict = {f"target_{j}": ys[:, j].numpy() for j in range(ys.shape[1])}
            
            test_df = pd.DataFrame({
                **outputs_dict,
                **targets_dict
            })
            test_df.to_csv(f'{test_path}/exit_{i}.csv', index=False)
            
            # Compute and log metrics
            avg_auc, avg_f1, avg_acc, avg_recall, aucs, f1_scores, accuracies, recall = compute_metrics(outputs, ys, self.best_threshold)
            self.log(f'test_avg_auc_{i}', avg_auc)
            self.log(f'test_avg_f1_{i}', avg_f1)
            self.log(f'test_avg_acc_{i}', avg_acc)

            entropy = custom_entropy_formula(outputs.detach().cpu().numpy())
            self.log(f'test_avg_entropy_{i}', entropy)
        
        # Save model values        
        values = {
            "best_auc": float(self.best_auc),
            "best_threshold": float(self.best_threshold),
            "exits" : {exit: self.exits_used[exit] for exit in range(len(self.exits_used))}
        }
        full_path = os.path.join(self.trainer.default_root_dir)
        with open(f'{full_path}/values.json', 'w') as f:
            json.dump(values, f, indent=4)

        self.test_outputs.clear()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def predict_step(self, batch):
        inputs, target = batch
        return self.model(inputs, target)
