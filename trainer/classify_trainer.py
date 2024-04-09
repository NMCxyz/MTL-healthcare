import os
import wandb
import torch
from tqdm import tqdm

from utils import save_json
from utils import pretty_print_json
from trainer.base_trainer import BaseTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClassifyTrainer(BaseTrainer):
    
    def __init__(
        self, model, train_dataset, eval_dataset,
        optimizer, batch_size, epochs, output_dir,
        log_steps, log_wandb, project_name, experiment_name,
        cls_loss_fn, cls_metric
    ):
        super().__init__(
            model, train_dataset, eval_dataset, optimizer,
            batch_size, epochs, output_dir, log_steps,
            log_wandb, project_name, experiment_name
        )
        self.cls_loss_fn = cls_loss_fn
        self.cls_metric = cls_metric
        
    def train(self):
        # Evaluate before training
        test_log = self.evaluate(
            test_dataloader=self.test_dataloader,
            model=self.model,
            compute_cls_loss=self.cls_loss_fn,
            cls_metric=self.cls_metric,
        )
        max_f1 = test_log["Test F1"]
        min_loss = test_log["Test Loss"]
        print("Evaluate before training:", flush=True)
        pretty_print_json(test_log)

        best_cls_log = {}
        best_loss_log = {}
        
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
                optimizer=self.optimizer,
                cls_metric=self.cls_metric,
            )
            test_log = self.evaluate(
                    test_dataloader=self.test_dataloader,
                    model=self.model,
                    compute_cls_loss=self.cls_loss_fn,
                    cls_metric=self.cls_metric,
                    train_log=train_log
            )

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
                
            # Save best loss checkpoint 
            if test_log["Test Loss"] > min_loss:
                patient += 1
                log_message = self.get_log_message(
                    epoch=epoch,
                    metric= "Test loss",
                    before=round(min_loss, 4),
                    after=round(test_log["Test Loss"], 4),
                    patient=True
                )
                tqdm.write(log_message, end="\n\n")
            else:
                patient = 0
                # Update record loss
                min_loss = test_log["Test Loss"]
                best_loss_log = test_log
                
                checkpoint_path = os.path.join(
                    self.output_dir,
                    "best_loss.pth"
                )
                self.save_checkpoint(checkpoint_path=checkpoint_path)
        #     if patient > 100:
        #         print(f"Early stopping at epoch {epoch + 1}!")
        #         break
            records = {
                "max_f1_test": round(max_f1, 2),
                "min_loss_test": round(min_loss, 2),
                "train_loss": round(test_log["Train Loss"], 2),
            }  
            pbar.set_postfix(**records)

        result_training = {
            "best_cls_log": best_cls_log,
            "best_loss_log": best_loss_log
        }
        log_path = os.path.join(self.output_dir, "result_training.json") 
        save_json(
            data=result_training,
            file_path=log_path
        )


    @staticmethod
    def _inner_training_loop(
            train_dataloader,
            model,
            cls_loss_fn,
            optimizer,
            cls_metric,
        ):
        num_batches = len(train_dataloader)
        
        total_loss_cls = 0
        total_acc = 0
        total_f1 = 0
        model.train()
        step = 0
        for x, y_cls in train_dataloader:
            cls_output = model(x)
            y_cls = y_cls.view(-1)
            cls_loss = cls_loss_fn(cls_output, y_cls)
            
            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step()
            
            
            total_loss_cls += cls_loss.item()
            acc, f1 = cls_metric(cls_output, y_cls)
            total_acc += acc
            total_f1 += f1
            step += 1
        avg_loss_cls = total_loss_cls / num_batches
        
        avg_acc = total_acc / num_batches
        avg_f1 = total_acc / num_batches
        
        log_result = {
            "Train Loss": avg_loss_cls,
            "Train Acc": avg_acc,
            "Train F1": avg_f1
        }
        
        return log_result

    @staticmethod
    def evaluate(
        test_dataloader,
        model,
        compute_cls_loss, 
        cls_metric,
        train_log={},
    ):
        num_batches = len(test_dataloader)
        total_loss_cls = 0
        
        total_acc = 0
        total_f1 = 0
        
        model.eval()
        with torch.no_grad():
            for x, y_cls in test_dataloader:
                y_cls = y_cls.view(-1)
                cls_output = model(x)
                cls_loss = compute_cls_loss(cls_output, y_cls)
                total_loss_cls += cls_loss.item()

                acc, f1 = cls_metric(cls_output, y_cls)
                total_acc += acc
                total_f1 += f1
                
                

        
        avg_loss_cls = total_loss_cls / num_batches
        avg_acc = total_acc / num_batches
        avg_f1 = total_acc / num_batches
        
        log_result = {
            "Test Loss": avg_loss_cls,
            "Test Acc": avg_acc,
            "Test F1": avg_f1
        }
        
        for k, v in log_result.items():
            log_result[k] = round(v, 4)
        log_result.update(train_log)
        return log_result