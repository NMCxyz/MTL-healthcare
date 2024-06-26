import argparse
import random
import itertools

import numpy as np
import torch

from dataset import get_data_mtl
from dataset import MultitaskDataset
from trainer.multitask_orthogonal_tracenorm_trainer import MultitaskOrthogonalTracenormTrainer
from net import (
    MultitaskLSTM,
    MultitaskGRU,
    MultitaskRNN,
    MultitaskMLSTMfcn,
    MultitaskTCN,
    cls_metric,
    cls_loss_fn,
    reg_loss_fn,
    reg_metric
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Hyper parameters for training")
    # Hyper parameter for training
    parser.add_argument('--batch_size', type=int, help='Batch size for training')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--input_dim', type=int, help='Input dimension')
    parser.add_argument('--n_hidden_1', type=int, help='Number of hidden units in the LSTM layer')
    parser.add_argument('--n_hidden_2', type=int, help='Number of hidden units in the LSTM layer')
    parser.add_argument('--n_classes', type=int, help='Number of output classes')
    parser.add_argument('--p_dropout', type=float, help='Dropout probability')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Set the random seed')
    parser.add_argument('--log_steps', type=int, help='Logging steps during training')
    parser.add_argument('--w_regression', type=float, default=1, help='Weight regression loss')
    parser.add_argument('--w_classify', type=float, default=1, help='Weight classify loss')
    parser.add_argument('--w_grad', type=float, default=1, help='Weight gradient loss')
    parser.add_argument('--w_trace_norm', type=float, default=1, help='Weight gradient loss')

    # Location of data and checkpoint 
    parser.add_argument('--data_path', type=str, help='Path to the data training')
    parser.add_argument('--output_dir', type=str, help='Output directory for saving models')

    # WandDB logging
    parser.add_argument('--log_wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--project_name', type=str, default='Project demo', help='WandB project name')
    parser.add_argument('--experiment_name', type=str, default='Experiment demo', help='WandB experiment name')
    
    # Model type
    parser.add_argument('--model_type', type=str, help='Type of model to use')
    
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

def train_model(args, w_regression, w_classify, w_grad, w_trace_norm):
    if args.seed:
        print("Initialize parameters with random seed: ", args.seed)
        set_random_seed(seed=args.seed)
    else:
        print("Initialize parameters random without seed")

    # Load data
    data = np.load(args.data_path)
    print("Loading data from {}...".format(args.data_path))
    tensor_data = get_data_mtl(data=data)
    train_dataset = MultitaskDataset(
        features=tensor_data["x_train"],
        cls_target=tensor_data["y_train_cls"],
        reg_target=tensor_data["y_train_reg"]
    )
    test_dataset = MultitaskDataset(
        features=tensor_data["x_test"],
        cls_target=tensor_data["y_test_cls"],
        reg_target=tensor_data["y_test_reg"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.model_type == "MultitaskLSTM":
        model = MultitaskLSTM(
            input_size=args.input_dim,
            hidden_size_1=args.n_hidden_1,
            hidden_size_2=args.n_hidden_2,
            output_size=args.n_classes,
            dropout=args.p_dropout
        )
    elif args.model_type == "MultitaskRNN":
        model = MultitaskRNN(
            input_size=args.input_dim,
            hidden_size_1=args.n_hidden_1,
            hidden_size_2=args.n_hidden_2,
            output_size=args.n_classes,
            dropout=args.p_dropout
        )
    elif args.model_type == "MultitaskGRU":
        model = MultitaskGRU(
            input_size=args.input_dim,
            hidden_size_1=args.n_hidden_1,
            hidden_size_2=args.n_hidden_2,
            output_size=args.n_classes,
            dropout=args.p_dropout
        )
    # Add more elif blocks here for other model types as needed
    else:
        raise ValueError("Unsupported model type: {}".format(args.model_type))
    
    model = model.to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.learning_rate
    )

    output_dir = f"{args.output_dir}/{w_regression}-{w_classify}-{w_grad}-{w_trace_norm}"
    
    print("Training info:\n")
    print("- Train data: {} samples".format(len(train_dataset)))
    print("- Test data: {} samples".format(len(test_dataset)))
    print("- Batch size: {}".format(args.batch_size))
    print("- Number of epochs: {}".format(args.epochs))
    print("- Learning rate: {}".format(args.learning_rate))
    print("Model config:\n", model)
    
    trainer = MultitaskOrthogonalTracenormTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        cls_loss_fn=cls_loss_fn,
        reg_loss_fn=reg_loss_fn,
        cls_metric=cls_metric,
        reg_metric=reg_metric,
        optimizer=optimizer,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=output_dir,
        log_steps=args.log_steps,
        log_wandb=args.log_wandb,
        project_name=args.project_name,
        experiment_name=f"mtl-LSTM-128-64-orthogonal-{w_regression}-{w_classify}-{w_grad}-{w_trace_norm}",
        weight_regression=w_regression,
        weight_classify=w_classify,
        weight_grad=w_grad,
        weight_trace_norm=w_trace_norm
    )
    
    trainer.train()

if __name__ == "__main__":
    args = parse_arguments()

    # w_regression_values = [0.4, 0.6]
    # w_classify_values = [0.9, 0.6, 0.5]
    # w_grad_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # w_trace_norm_values = [0.1, 0.01, 0.001]
    
    w_regression_values = [0.4]
    w_classify_values = [0.9]
    w_grad_values = [0.3]
    w_trace_norm_values = [0.001]

    combinations = list(itertools.product(w_regression_values, w_classify_values, w_grad_values, w_trace_norm_values))

    for w_regression, w_classify, w_grad, w_trace_norm in combinations:
        print(f"Training with w_regression={w_regression}, w_classify={w_classify}, w_grad={w_grad}, w_trace_norm={w_trace_norm}")
        train_model(args, w_regression, w_classify, w_grad, w_trace_norm)
