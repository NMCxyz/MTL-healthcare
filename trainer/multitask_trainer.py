import os
import wandb
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm
sns.set()
from utils import save_json
from utils import pretty_print_json
from trainer.base_trainer import BaseTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultitaskTrainer(BaseTrainer):
    
    def __init__(
        self, model, train_dataset, eval_dataset,
        optimizer, batch_size, epochs, output_dir,
        log_steps, log_wandb, project_name, experiment_name,
        cls_loss_fn, reg_loss_fn, cls_metric, reg_metric
    ):
        super().__init__(
            model, train_dataset, eval_dataset, optimizer,
            batch_size, epochs, output_dir, log_steps,
            log_wandb, project_name, experiment_name
        )
        self.cls_loss_fn = cls_loss_fn
        self.reg_loss_fn = reg_loss_fn
        self.cls_metric = cls_metric
        self.reg_metric = reg_metric
        

    def train(self):
        self.history_training = {
            "train": {
                "Train Loss": [],
                "Train Loss Reg": [],
                "Train Loss Cls": [],
                "Train MAE": [],
                "Train Acc": [],
                "Train F1": []
            },
            "test": {
                "Test Loss": [],
                "Test Loss Reg": [],
                "Test Loss Cls": [],
                "Test MAE": [],
                "Test Acc": [],
                "Test F1": []
            }
        }
        # Evaluate before training
        test_log = self.evaluate(
            test_dataloader=self.test_dataloader,
            model=self.model,
            compute_cls_loss=self.cls_loss_fn,
            compute_reg_loss=self.reg_loss_fn,
            cls_metric=self.cls_metric,
            reg_metric=self.reg_metric
        )
        max_f1 = test_log["Test F1"]
        min_mae = test_log["Test MAE"]
        min_loss = test_log["Test Loss"]
        print("Evaluate before training:", flush=True)
        pretty_print_json(test_log)

        best_cls_log = {}
        best_reg_log = {}
        best_multitask_log = {}
        
        # Training
        self.model.to(device)
        patient = 0
        pbar = tqdm(
            range(self.epochs),
            ncols=180,
            colour="green",
            desc="Training"
        )
        for epoch in pbar:
            train_log = self._inner_training_loop(
                train_dataloader=self.train_dataloader,
                model=self.model, 
                cls_loss_fn=self.cls_loss_fn,
                reg_loss_fn=self.reg_loss_fn,
                optimizer=self.optimizer,
                cls_metric=self.cls_metric,
                reg_metric=self.reg_metric
            )
            for k in self.history_training["train"]:
                self.history_training["train"][k].append(train_log[k])
            test_log = self.evaluate(
                    test_dataloader=self.test_dataloader,
                    model=self.model,
                    compute_cls_loss=self.cls_loss_fn,
                    compute_reg_loss=self.reg_loss_fn,
                    cls_metric=self.cls_metric,
                    reg_metric=self.reg_metric,
                    train_log=train_log
            )
            for k in self.history_training["test"]:
                self.history_training["test"][k].append(test_log[k])
            if self.log_wandb:
                wandb.log(test_log)

            # Save best checkpoint classify
            if test_log["Test F1"] > max_f1:
                log_message = self.get_log_message(
                    epoch=epoch,
                    metric="Test F1",
                    before=round(max_f1, 4),
                    after=round(test_log["Test F1"], 4)
                )
                tqdm.write(log_message, end="\n\n")
                # Update record classify metric
                max_f1 = test_log["Test F1"]
                best_cls_log = test_log

                checkpoint_path = os.path.join(
                    self.output_dir,
                    "best_cls.pth"
                )
                self.save_checkpoint(checkpoint_path=checkpoint_path)
                
            # Save best checkpoint regression
            if test_log["Test MAE"] < min_mae:
                log_message = self.get_log_message(
                    epoch=epoch,
                    metric="Test MAE",
                    before=round(min_mae, 4),
                    after=round(test_log["Test MAE"], 4)
                )
                tqdm.write(log_message, end="\n\n")
                # Update record regression metric 
                min_mae = test_log["Test MAE"]
                best_reg_log = test_log

                checkpoint_path = os.path.join(
                    self.output_dir,
                    "best_reg.pth"
                )
                self.save_checkpoint(checkpoint_path=checkpoint_path)
                
            # Save best multitask checkpoint 
            if test_log["Test Loss"] > min_loss:
                patient += 1
                log_message = self.get_log_message(
                    epoch=epoch,
                    metric="Multitask test loss",
                    before=round(min_loss, 4),
                    after=round(test_log["Test Loss"], 4),
                    patient=True
                )
                tqdm.write(log_message, end="\n\n")
            else:
                patient = 0
                # Update record multitask loss
                min_loss = test_log["Test Loss"]
                best_multitask_log = test_log
                
                checkpoint_path = os.path.join(
                    self.output_dir,
                    "best_multitask.pth"
                )
                self.save_checkpoint(checkpoint_path=checkpoint_path)
        #     if patient > 100:
        #         print(f"Early stopping at epoch {epoch + 1}!")
        #         break
            records = {
                "max_f1_test": round(max_f1, 2),
                "min_mae_test": round(min_mae, 2),
                "min_loss_test": round(min_loss, 2),
                "train_loss": round(test_log["Train Loss"], 2),
                "train_mse": round(test_log["Train Loss Reg"], 2),
                "train_ce": round(test_log["Train Loss Cls"], 2)
            }  
            pbar.set_postfix(**records)

        result_training = {
            "best_metrics": {
                "best_cls_log": best_cls_log,
                "best_reg_log": best_reg_log,
                "best_multitask_log": best_multitask_log
            }
        }
        log_path = os.path.join(self.output_dir, "result_training.json") 
        save_json(
            data=result_training,
            file_path=log_path
        )

        history_path = os.path.join(self.output_dir, "history_training.json")
        save_json(
            data=self.history_training, 
            file_path=history_path
        )

        self.visualize_history_training(
            history_metrics=self.history_training["train"],
            save_path=os.path.join(self.output_dir, "train_log.png"),
            title="Training history",
            n_epochs=self.epochs
        )
        self.visualize_history_training(
            history_metrics=self.history_training["test"],
            save_path=os.path.join(self.output_dir, "validate_log.png"),
            title="Validate history",
            n_epochs=self.epochs
        )

    def _inner_training_loop(
            self,
            train_dataloader,
            model,
            cls_loss_fn,
            reg_loss_fn,
            optimizer,
            cls_metric,
            reg_metric
        ):
        num_batches = len(train_dataloader)
        total_loss = 0
        
        total_loss_reg = 0
        total_loss_cls = 0
        
        total_mae = 0
        total_acc = 0
        total_f1 = 0
        model.train()
        step = 0
        for x, y_cls, y_reg in train_dataloader:
            reg_output, cls_output = model(x)
            reg_loss = reg_loss_fn(reg_output, y_reg)
            y_cls = y_cls.view(-1)
            cls_loss = cls_loss_fn(cls_output, y_cls)
            
            loss = reg_loss + cls_loss + reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            total_loss_reg += reg_loss.item()
            total_loss_cls += cls_loss.item()
            total_loss += loss.item()
            
            total_mae += reg_metric(reg_output, y_reg).item()
            acc, f1 = cls_metric(cls_output, y_cls)
            total_acc += acc
            total_f1 += f1
            step += 1
        avg_loss = total_loss / num_batches
        avg_loss_cls = total_loss_cls / num_batches
        avg_loss_reg = total_loss_reg / num_batches
        
        avg_mae = total_mae / num_batches
        avg_acc = total_acc / num_batches
        avg_f1 = total_acc / num_batches
        
        log_result = {
            "Train Loss": avg_loss,
            "Train Loss Reg": avg_loss_reg,
            "Train Loss Cls": avg_loss_cls,
            "Train MAE": avg_mae,
            "Train Acc": avg_acc,
            "Train F1": avg_f1
        }
        
        return log_result

    @staticmethod
    def evaluate(
        test_dataloader,
        model,
        compute_cls_loss, 
        compute_reg_loss,
        cls_metric,
        reg_metric,
        train_log={},
    ):
        num_batches = len(test_dataloader)
        total_loss = 0
        total_loss_reg = 0
        total_loss_cls = 0
        
        total_mae = 0
        total_acc = 0
        total_f1 = 0
        
        model.eval()
        with torch.no_grad():
            for x, y_cls, y_reg in test_dataloader:
                y_cls = y_cls.view(-1)
                reg_output, cls_output = model(x)
                reg_loss = compute_reg_loss(reg_output, y_reg)
                cls_loss = compute_cls_loss(cls_output, y_cls)
                
                loss = reg_loss + cls_loss

                total_loss_cls += cls_loss.item()
                total_loss_reg += reg_loss.item()

                total_loss += loss.item()
                total_mae += reg_metric(reg_output, y_reg).item()
                acc, f1 = cls_metric(cls_output, y_cls)
                total_acc += acc
                total_f1 += f1
                
                

        avg_loss = total_loss / num_batches
        
        avg_loss_cls = total_loss_cls / num_batches
        avg_loss_reg = total_loss_reg / num_batches
        avg_mae = total_mae / num_batches
        avg_acc = total_acc / num_batches
        avg_f1 = total_acc / num_batches
        
        log_result = {
            "Test Loss": avg_loss,
            "Test Loss Reg": avg_loss_reg,
            "Test Loss Cls": avg_loss_cls,
            "Test MAE": avg_mae,
            "Test Acc": avg_acc,
            "Test F1": avg_f1
        }
        
    #     log_result = {
    #         "Test Loss": avg_loss,
    #         "Test Acc": avg_acc,
    #         "Test F1": avg_f1
    #     }
        for k, v in log_result.items():
            log_result[k] = round(v, 4)
        log_result.update(train_log)
        return log_result
    
    def visualize_history_training(self, history_metrics, save_path, title, n_epochs):
        all_colors = [
            [0.6741758166170712, 0.22868885920931392, 0.2406746141609114],
            [0.059507814505031176, 0.792092060272692, 0.7892093337887586],
            [0.3463890703039292, 0.6354515889226895, 0.9495167990408476],
            [0.6590696515728108, 0.09209113056341345, 0.572954686134253],
            [0.6000209199288403, 0.5327399348368033, 0.8311295309445689],
            [0.1695333359497634, 0.6360509540985432, 0.13766664393429084]
        ]
        epochs = list(range(1, n_epochs + 1))
        metrics = list(history_metrics.keys())
        num_metrics = len(metrics)
        ncols = 3
        nrows = 2
        fig, axes = plt.subplots(nrows, ncols, figsize=(30, nrows * 5))
        fig.suptitle(title, fontsize=16)
        # Plotting training and testing metrics
        for idx, metric in enumerate(history_metrics):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row, col]
            ax.plot(epochs, history_metrics[metric], marker='o', label='Training', color=all_colors[idx])
            ax.set_xlabel('Epochs')
            ax.set_ylabel(metric)
            ax.grid(True)

        fig.savefig(save_path)
        plt.close(fig)