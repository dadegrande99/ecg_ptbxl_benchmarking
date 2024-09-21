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
        self.dropout = nn.Dropout(p=dropout_rate)
        self.weights_ee = []
        self.test_all_exits = True
        self.exits_used = []
        self.all_results = {}

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Forward pass of the model
        Is composed of three main parts:
        - forward_intro: Introduction of the model (e.g. Convolutional layers)
        - forward_modules: Main part of the model (e.g. Residual blocks)
        - forward_final: Final part of the model (e.g. Fully connected layers)

        Parameters:
        - x (torch.Tensor): Input tensor to the model

        Returns:
        - tuple: Tuple containing the outputs of the model
        """
        x = self.forward_intro(x)

        x, outputs = self.forward_modules(x)

        if len(outputs) <= len(self.modules_EE):
            outputs.append(self.forward_final(x))

        return tuple(outputs)

    @abstractmethod
    def forward_intro(self, x: torch.Tensor) -> torch.Tensor:
        """
        Introduction of the model (e.g. Convolutional layers)

        Parameters:
        - x (torch.Tensor): Input tensor to the model

        Returns:
        - torch.Tensor: Output tensor of the introduction part"""
        pass

    def forward_modules(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Main part of the model, composed of multiple modules and exits with dropout layers

        Parameters:
        - x (torch.Tensor): Input tensor to the model

        Returns:
        - tuple: Tuple containing the outputs of the model
        """
        outputs = []
        exit_used = -1

        for i, layer in enumerate(self.modules_EE):
            x = layer(x)
            x = self.dropout(x)
            # debug dimensione
            # out = x #
            out = x.mean(dim=2)
            # print(out.shape)
            # dropout

            outputs.append(torch.sigmoid(self.exits[i](out)))
            should_exit = self.should_exit(outputs[-1], self.entropy_threshold)
            # possibility to calculate the loss for the exit???

            if should_exit and exit_used == -1:
                exit_used = i
            if not (self.training) and should_exit:
                for _ in range(i, len(self.exits)):
                    outputs.append(outputs[-1])
                self.exits_used[i] += 1
                return x, outputs
        self.exits_used[exit_used] += 1
        return x, outputs

    @abstractmethod
    def forward_final(self, x: torch.Tensor) -> torch.Tensor:
        """
        Final part of the model, composed of fully connected layers

        Parameters:
        - x (torch.Tensor): Input tensor to the model

        Returns:
        - torch.Tensor: Output tensor of the final part
        """

        pass

    def should_exit(self, predictions: torch.Tensor, threshold: float) -> bool:
        """
        Check if the model should exit based on the entropy of the predictions

        Parameters:
        - predictions (torch.Tensor): Predictions of the model
        - threshold (float): Threshold for the entropy

        Returns:
        - bool: Whether the model should exit or not
        """
        predictions_np = predictions.detach().cpu().numpy()
        entropy = custom_entropy_formula(predictions_np)
        return entropy < threshold

    def turn_on_dropout(self) -> None:
        """
        Turn on the dropout layer

        Returns:
        - None
        """
        self.dropout = nn.Dropout(p=self.dropout_rate)

    def turn_off_dropout(self) -> None:
        """
        Turn off the dropout layer

        Returns:
        - None
        """
        self.dropout = nn.Dropout(p=0)

    def on_train_start(self) -> None:
        """
        Turn off the dropout layer at the beginning of the training

        Returns:
        - None
        """
        self.turn_off_dropout()
        return super().on_train_start()

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Training step of the model, composed of the forward pass and the loss calculation

        Parameters:
        - batch (tuple): Tuple containing the inputs and the targets
        - batch_idx (int): Index of the batch

        Returns:
        - torch.Tensor: Total loss of the model
        """
        inputs, target = batch
        outputs = self(inputs)
        total_loss = self.loss(outputs, target, self.weights_ee)
        self.log('train_loss', total_loss)
        self.training_outputs.append((outputs, target))
        return total_loss

    def on_train_epoch_end(self) -> None:
        """
        End of the training epoch, save the outputs and compute the metrics

        Returns:
        - None
        """
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
            exit_path = os.path.join(
                self.trainer.default_root_dir, f'train/exit_{i}')
            os.makedirs(exit_path, exist_ok=True)

            # Save the outputs and targets
            outputs_dict = {f"output_{j}": outputs[:, j].numpy(
            ) for j in range(outputs.shape[1])}
            targets_dict = {
                f"target_{j}": ys[:, j].numpy() for j in range(ys.shape[1])}

            train_df = pd.DataFrame({
                **outputs_dict,
                **targets_dict
            })
            train_df.to_csv(
                f'{exit_path}/{self.current_epoch:03d}.csv', index=False)

            # Compute and log metrics
            avg_auc, avg_f1, avg_acc, avg_recall, aucs, f1_scores, accuracies, recall = compute_metrics(
                outputs, ys, self.best_threshold)
            entropy = custom_entropy_formula(outputs.detach().cpu().numpy())
            self.all_results["train"][f"exit_{i}"]["auc"].append(avg_auc)
            self.all_results["train"][f"exit_{i}"]["f1"].append(avg_f1)
            self.all_results["train"][f"exit_{i}"]["acc"].append(avg_acc)
            self.all_results["train"][f"exit_{i}"]["recall"].append(avg_recall)
            self.all_results["train"][f"exit_{i}"]["entropy"].append(entropy)
            for key in self.all_results["train"][f"exit_{i}"]:
                self.log(f'train_{key}_{i}',
                         self.all_results["train"][f"exit_{i}"][key][-1])

        self.training_outputs.clear()

    def on_validation_start(self) -> None:
        """
        Turn on the dropout layer at the beginning of the validation

        Returns:
        - None
        """
        self.turn_on_dropout()
        return super().on_validation_start()

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Validation step of the model, composed of the forward pass and the loss calculation

        Parameters:
        - batch (tuple): Tuple containing the inputs and the targets
        - batch_idx (int): Index of the batch

        Returns:
        - torch.Tensor: Total loss of the model
        """
        inputs, target = batch
        outputs = self(inputs)
        total_loss = self.loss(outputs, target, self.weights_ee)
        self.log("val_loss", total_loss)
        self.validation_outputs.append((outputs, target))
        return total_loss

    def on_validation_epoch_end(self) -> None:
        """
        End of the validation epoch, save the outputs and compute the metrics

        Returns:
        - None
        """
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
            exit_path = os.path.join(
                self.trainer.default_root_dir, f'val/exit_{i}')
            os.makedirs(exit_path, exist_ok=True)

            # Save the outputs and targets
            outputs_dict = {f"output_{j}": outputs[:, j].numpy(
            ) for j in range(outputs.shape[1])}
            targets_dict = {f"target_{j}": targets[:, j].numpy(
            ) for j in range(targets.shape[1])}

            val_df = pd.DataFrame({
                **outputs_dict,
                **targets_dict
            })
            val_df.to_csv(
                f'{exit_path}/{self.current_epoch:03d}.csv', index=False)

            # Compute and log metrics
            best_exit = {
                "avg_auc": 0.0,
                "avg_f1": 0.0,
                "avg_acc": 0.0
            }
            for threshold in self.thresholds:
                avg_auc, avg_f1, avg_acc, avg_recall, aucs, f1_scores, accuracies, recall = compute_metrics(
                    outputs, targets, threshold)  # type: ignore
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

            self.all_results["val"][f"exit_{i}"]["auc"].append(
                best_exit["avg_auc"])
            self.all_results["val"][f"exit_{i}"]["f1"].append(
                best_exit["avg_f1"])
            self.all_results["val"][f"exit_{i}"]["acc"].append(
                best_exit["avg_acc"])
            self.all_results["val"][f"exit_{i}"]["recall"].append(avg_recall)
            entropy = custom_entropy_formula(outputs.detach().cpu().numpy())
            self.all_results["val"][f"exit_{i}"]["entropy"].append(entropy)

            for key in self.all_results["val"][f"exit_{i}"]:
                self.log(f'val_{key}_{i}',
                         self.all_results["val"][f"exit_{i}"][key][-1])

        self.log(f'val_avg_auc', best_epoch["avg_auc"])
        self.log(f'val_avg_f1', best_epoch["avg_f1"])
        self.log(f'val_avg_acc', best_epoch["avg_acc"])

        self.validation_outputs.clear()

    def on_test_start(self) -> None:
        """
        Turn on the dropout layer at the beginning of the test

        Returns:
        - None
        """
        self.turn_on_dropout()
        return super().on_test_start()

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """
        Test step of the model, composed of the forward pass and the loss calculation

        Parameters:
        - batch (tuple): Tuple containing the inputs and the targets
        - batch_idx (int): Index of the batch

        Returns:
        - torch.Tensor: Total loss of the model
        """
        inputs, target = batch
        outputs = self(inputs)
        total_loss = self.loss(outputs, target, self.weights_ee)
        self.log("test_loss", total_loss)
        self.test_outputs.append((outputs, target))
        return total_loss

    def on_test_epoch_end(self) -> None:
        """
        End of the test epoch, save the outputs and compute the metrics

        Returns:
        - None
        """
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
            outputs_dict = {f"output_{j}": outputs[:, j].numpy(
            ) for j in range(outputs.shape[1])}
            targets_dict = {
                f"target_{j}": ys[:, j].numpy() for j in range(ys.shape[1])}

            test_df = pd.DataFrame({
                **outputs_dict,
                **targets_dict
            })
            test_df.to_csv(f'{test_path}/exit_{i}.csv', index=False)

            # Compute and log metrics
            avg_auc, avg_f1, avg_acc, avg_recall, aucs, f1_scores, accuracies, recall = compute_metrics(
                outputs, ys, self.best_threshold)
            entropy = custom_entropy_formula(outputs.detach().cpu().numpy())
            self.all_results["test"][f"exit_{i}"]["auc"].append(avg_auc)
            self.all_results["test"][f"exit_{i}"]["f1"].append(avg_f1)
            self.all_results["test"][f"exit_{i}"]["acc"].append(avg_acc)
            self.all_results["test"][f"exit_{i}"]["recall"].append(avg_recall)
            self.all_results["test"][f"exit_{i}"]["entropy"].append(entropy)
            for key in self.all_results["test"][f"exit_{i}"]:
                self.log(f'test_{key}_{i}',
                         self.all_results["test"][f"exit_{i}"][key][-1])

        self.test_outputs.clear()

    def configure_optimizers(self):
        """
        Configure the optimizer for the model

        Returns:
        - torch.optim.Optimizer: Optimizer for the model
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)

    def predict_step(self, batch: tuple, batch_idx: int, dataloader_idx: int = None) -> torch.Tensor:
        """
        Predict step of the model, composed of the forward pass

        Parameters:
        - batch (tuple): Tuple containing the inputs and the targets
        - batch_idx (int): Index of the batch
        - dataloader_idx (int): Index of the dataloader

        Returns:
        - torch.Tensor: Predictions of the model
        """
        inputs, target = batch
        return self.model(inputs, target)

    def mcd_validation(self, mcd_loader, targets=None, num_tests: int = 5, save: bool = True, output_dir: str = "mcd_validation") -> tuple[dict, dict]:
        """
        Perform Monte Carlo Dropout validation by running multiple forward passes through the model with dropout enabled to estimate prediction uncertainties

        Parameters:
        - mcd_loader (DataLoader): DataLoader for Monte Carlo Dropout validation.
        - targets (np.ndarray): Targets for the data. If None, targets will be extracted from the DataLoader.
        - num_tests (int): Number of tests to run for Monte Carlo Dropout validation. Default is 5.
        - save (bool): Whether to save the outputs to a CSV file. Default is True.
        - output_dir (str): Directory to save the outputs. Default is 'mcd_validation'.

        Returns:
        - tuple[dict, dict]: A tuple containing the results and the outputs as DataFrames.
            - dict: Results of the Monte Carlo Dropout validation.
            - dict: Outputs of the Monte Carlo Dropout validation as DataFrames.
        """

        if mcd_loader is None:
            raise ValueError("DataLoader is None.")
        if targets is None:
            # get targets from mcd_loader
            targets = []
            for _, data in enumerate(mcd_loader):
                _, target = data
                targets.append(target)
            targets = np.concatenate(targets, axis=0)

        if output_dir is None:
            output_dir = "mcd_validation"
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(
                self.trainer.default_root_dir, output_dir)
        dir_name = os.path.basename(output_dir)

        # progess bar
        other_loop = 4
        pbar = tqdm(total=num_tests * len(mcd_loader) + other_loop,
                    desc="MonteCarloDropout Validation Progress")

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

        outputs_df = {}
        for key in outputs:
            num_tests = len(outputs[key])
            # Stack outputs to shape (N, D, num_tests)
            # Shape: (N, D, num_tests)
            outputs_array = np.stack(outputs[key], axis=-1)
            N, D, num_tests = outputs_array.shape

            # Prepare data for DataFrame
            data = []
            for n in range(N):
                row = {}
                for d in range(D):
                    # Collect outputs across tests for data point n, dimension d
                    # Shape: (num_tests,)
                    outputs_list = outputs_array[n, d, :]
                    outputs_list = outputs_list.tolist()    # Convert to list
                    # Assign to the appropriate column
                    row[f'output_{d+1}'] = outputs_list
                # Add target values
                for t in range(targets.shape[1]):
                    row[f'target_{t+1}'] = targets[n, t]
                data.append(row)
            # Create DataFrame
            outputs_df[f'exit_{key}'] = pd.DataFrame(data)
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
            avg_auc, avg_f1, avg_acc, recall = compute_metrics(
                outputs_i, targets, self.best_threshold)[:4]
            entropy = custom_entropy_formula(outputs_i)
            self.all_results[dir_name][f"exit_{i}"]["auc"].append(avg_auc)
            self.all_results[dir_name][f"exit_{i}"]["f1"].append(avg_f1)
            self.all_results[dir_name][f"exit_{i}"]["acc"].append(avg_acc)
            self.all_results[dir_name][f'exit_{i}']["recall"].append(recall)
            self.all_results[dir_name][f"exit_{i}"]["entropy"].append(entropy)
            # log is not supported
        pbar.update(1)

        if save:  # Save outputs
            os.makedirs(output_dir, exist_ok=True)
            for key in outputs_df:
                outputs_df[key].to_csv(os.path.join(
                    output_dir, f"{key}.csv"), index=False)

        pbar.close()  # Close progress bar

        return (self.all_results[dir_name], outputs_df)

    def save_values(self, path=None, save: bool = True) -> dict:
        """
        Save the values of the model

        Parameters:
        - path (str): Path to save the values. Default is None.
        - save (bool): Whether to save the values. Default is True.

        Returns:
        - dict: Dictionary containing the values of the model
        """
        if path is None:
            path = self.trainer.default_root_dir
        values = {
            "best_auc": float(self.best_auc),
            "best_threshold": float(self.best_threshold),
            "exits": {exit: self.exits_used[exit] for exit in range(len(self.exits_used))}
        }
        if save:
            os.makedirs(path, exist_ok=True)
            with open(f'{path}/values.json', 'w') as f:
                json.dump(values, f, indent=4)

        return values

    def save_results(self, path: str = "results", save: bool = True) -> dict:
        """
        Save the results of the model

        Parameters:
        - path (str): Path to save the results. Default is 'results'.
        - save (bool): Whether to save the results. Default is True.

        Returns:
        - dict: Dictionary containing the results of the model
        """
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
