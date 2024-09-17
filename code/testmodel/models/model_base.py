from abc import ABC, abstractmethod
import json
from tqdm import tqdm
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
        self.all_results = {}

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
            x = self.dropout(x)
            ## debug dimensione
            # out = x # 
            out = x.mean(dim=2)
            # print(out.shape)
            # dropout

            outputs.append(torch.sigmoid(self.exits[i](out)))
            should_exit = self.should_exit(outputs[-1], self.entropy_threshold)
            ## possibility to calculate the loss for the exit???

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

        if "train" not in self.all_results:
            self.all_results["train"] = {}
            for i in range(len(y_hats[0])):
                self.all_results["train"][f"exit_{i}"] = {
                    "auc": [],
                    "f1": [],
                    "acc": [],
                    "recall": [],
                    "entropy": []
                }
        
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
            entropy = custom_entropy_formula(outputs.detach().cpu().numpy())
            self.all_results["train"][f"exit_{i}"]["auc"].append(avg_auc)
            self.all_results["train"][f"exit_{i}"]["f1"].append(avg_f1)
            self.all_results["train"][f"exit_{i}"]["acc"].append(avg_acc)
            self.all_results["train"][f"exit_{i}"]["recall"].append(avg_recall)
            self.all_results["train"][f"exit_{i}"]["entropy"].append(entropy)
            for key in self.all_results["train"][f"exit_{i}"]:
                self.log(f'train_{key}_{i}', self.all_results["train"][f"exit_{i}"][key][-1])
        
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

        if "val" not in self.all_results:
            self.all_results["val"] = {}
            for i in range(len(y_hats[0])):
                self.all_results["val"][f"exit_{i}"] = {
                    "auc": [],
                    "f1": [],
                    "acc": [],
                    "recall": [],
                    "entropy": []
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
            
            self.all_results["val"][f"exit_{i}"]["auc"].append(best_exit["avg_auc"])
            self.all_results["val"][f"exit_{i}"]["f1"].append(best_exit["avg_f1"])
            self.all_results["val"][f"exit_{i}"]["acc"].append(best_exit["avg_acc"])
            self.all_results["val"][f"exit_{i}"]["recall"].append(avg_recall)
            entropy = custom_entropy_formula(outputs.detach().cpu().numpy())
            self.all_results["val"][f"exit_{i}"]["entropy"].append(entropy)
            
            for key in self.all_results["val"][f"exit_{i}"]:
                self.log(f'val_{key}_{i}', self.all_results["val"][f"exit_{i}"][key][-1])

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

        if "test" not in self.all_results:
            self.all_results["test"] = {}
            for i in range(len(y_hats[0])):
                self.all_results["test"][f"exit_{i}"] = {
                    "auc": [],
                    "f1": [],
                    "acc": [],
                    "recall": [],
                    "entropy": []
                }
        
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
            entropy = custom_entropy_formula(outputs.detach().cpu().numpy())
            self.all_results["test"][f"exit_{i}"]["auc"].append(avg_auc)
            self.all_results["test"][f"exit_{i}"]["f1"].append(avg_f1)
            self.all_results["test"][f"exit_{i}"]["acc"].append(avg_acc)
            self.all_results["test"][f"exit_{i}"]["recall"].append(avg_recall)
            self.all_results["test"][f"exit_{i}"]["entropy"].append(entropy)
            for key in self.all_results["test"][f"exit_{i}"]:
                self.log(f'test_{key}_{i}', self.all_results["test"][f"exit_{i}"][key][-1])

        self.test_outputs.clear()

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def predict_step(self, batch):
        inputs, target = batch
        return self.model(inputs, target)
    
    def mcd_validation(self, mcd_loader, tagets=None, num_tests:int = 5, save:bool = True, output_dir:str = "mcd_validation")-> tuple[dict, dict]:

        if mcd_loader is None:
            raise ValueError("DataLoader is None.")
        if tagets is None:
            # get targets from mcd_loader
            targets = []
            for _, data in enumerate(mcd_loader):
                _, target = data
                targets.append(target)
            targets = np.concatenate(targets, axis=0)

        if output_dir is None:
            output_dir = "mcd_validation"
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(self.trainer.default_root_dir, output_dir)
        dir_name = os.path.basename(output_dir)


        ## progess bar
        other_loop = 4
        pbar = tqdm(total=num_tests * len(mcd_loader) + other_loop, desc="MonteCarloDropout Validation Progress")

        # Compute outputs
        outs = []
        for _ in range(num_tests):
            outputs = {}
            for _, data in enumerate(mcd_loader):
                inputs, target = data
                out_i = self(inputs)
                for j, out in enumerate(out_i):
                    if j not in outputs:
                        outputs[j] = []
                    outputs[j].append(out.detach().cpu().numpy())
                pbar.update(1)
            for key in outputs:
                outputs[key] = np.concatenate(outputs[key], axis=0)
            outs.append(outputs)

        # Calculate mean of outputs
        outputs = {}
        for el in outs:
            for key in el:
                if key not in outputs:
                    outputs[key] = []
                outputs[key].append(el[key])
        pbar.update(1)

        for key in outputs:
            outputs[key] = np.mean(outputs[key], axis=0)
        pbar.update(1)
        
        # Compue metrics
        if dir_name not in self.all_results:
            self.all_results[dir_name] = {}
            for i in range(len(outputs)):
                self.all_results[dir_name][f"exit_{i}"] = {
                    "auc": [],
                    "f1": [],
                    "acc": [],
                    "recall": [],
                    "entropy": []
                }
        for i in range(len(outputs)):
            outputs_i = outputs[i]
            avg_auc, avg_f1, avg_acc, avg_recall, aucs, f1_scores, accuracies, recall = compute_metrics(outputs_i, targets, self.best_threshold)
            entropy = custom_entropy_formula(outputs_i)
            self.all_results[dir_name][f"exit_{i}"]["auc"].append(avg_auc)
            self.all_results[dir_name][f"exit_{i}"]["f1"].append(avg_f1)
            self.all_results[dir_name][f"exit_{i}"]["acc"].append(avg_acc)
            self.all_results[dir_name][f"exit_{i}"]["recall"].append(avg_recall)
            self.all_results[dir_name][f"exit_{i}"]["entropy"].append(entropy)
            # log is not supported
        pbar.update(1)
                
        # Transofrm outputs & targets to DataFrame
        outputs_df = {}
        for key in outputs:
            outputs_df[f"exit_{key}"] = pd.DataFrame(np.hstack((outputs[key], targets)), 
                    columns=[f'output_{i+1}' for i in range(outputs[key].shape[1])] + 
                            [f'target_{i+1}' for i in range(targets.shape[1])])
        pbar.update(1)
   
        if save: # Save outputs
            os.makedirs(output_dir, exist_ok=True)
            for key in outputs_df:
                outputs_df[key].to_csv(os.path.join(output_dir, f"{key}.csv"), index=False)

        pbar.close()# Close progress bar

        return (self.all_results[dir_name], outputs_df)
    
    def save_values(self, path = None, save:bool = True):
        if path is None:
            path = self.trainer.default_root_dir
        values = {
            "best_auc": float(self.best_auc),
            "best_threshold": float(self.best_threshold),
            "exits" : {exit: self.exits_used[exit] for exit in range(len(self.exits_used))}
        }
        if save:
            os.makedirs(path, exist_ok=True)
            with open(f'{path}/values.json', 'w') as f:
                json.dump(values, f, indent=4)

        return values
    
    def save_results(self, path:str = "results", save:bool = True):
        if save:
            if path is None:
                path = self.trainer.default_root_dir
            if not os.path.isabs(path):
                path = os.path.join(self.trainer.default_root_dir, path)
            os.makedirs(path, exist_ok=True)
            for key in self.all_results:
                for exit in self.all_results[key]:

                    df = pd.DataFrame(self.all_results[key][exit])
                    df.to_csv(f'{path}/{key}_{exit}.csv', index=False)

        return self.all_results
