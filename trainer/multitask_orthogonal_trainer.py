import os
import wandb
import torch
from tqdm import tqdm
import numpy as np
from utils import save_json
from utils import pretty_print_json
from trace_norm import TensorTraceNorm
from trainer.multitask_trainer import MultitaskTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultitaskOrthogonalTrainer(MultitaskTrainer):

    def __init__(
            self, model, train_dataset, eval_dataset, \
            optimizer, batch_size, epochs, output_dir, \
            log_steps, log_wandb, project_name, experiment_name, \
            cls_loss_fn, reg_loss_fn, cls_metric, reg_metric, \
            weight_regression, weight_classify, weight_grad, weight_trace_norm
    ):
        super().__init__(
            model, train_dataset, eval_dataset, optimizer, batch_size, \
            epochs, output_dir, log_steps, log_wandb, project_name, \
            experiment_name, cls_loss_fn, reg_loss_fn, cls_metric, reg_metric
        )

        self.w_reg = weight_regression
        self.w_cls = weight_classify
        self.w_grad = weight_grad
        self.weight_trace_norm = weight_trace_norm

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

            grads_reg = torch.autograd.grad(reg_loss, model.rnn.parameters(), retain_graph=True)
            grads_cls = torch.autograd.grad(cls_loss, model.rnn.parameters(), retain_graph=True)

            trace_norm_regular_list = []
            for param in model.rnn.parameters():
                if len(param.shape) == 1:
                    continue
                trace_norm_regular = torch.mean(TensorTraceNorm(param))
                trace_norm_regular_list.append(trace_norm_regular)

            trace_norm_regular = torch.mean(torch.stack(trace_norm_regular_list))

            grad_loss = 0
            #
            for i in range(len(grads_reg)):
                grad_loss += torch.norm(
                    (torch.mul(grads_cls[i], grads_reg[i]) - torch.ones_like(grads_reg[i]).to(device)), 2
                )

            loss = self.w_reg * reg_loss + self.w_cls * cls_loss + self.w_grad * grad_loss \
                    + self.weight_trace_norm*trace_norm_regular

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
