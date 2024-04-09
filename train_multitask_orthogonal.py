import argparse
import random

import numpy as np
import torch

from dataset import get_data_mtl
from dataset import MultitaskDataset
# from trainer import MultitaskTrainer
from trainer.multitask_orthogonal_trainer import MultitaskOrthogonalTrainer
from net import (
    MultitaskRNN,
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
    parser.add_argument('--fix_random', type=bool, default=False, help='Learning rate')
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
    
    args = parser.parse_args()
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_arguments()

    # set random seed
    if args.fix_random:
        print("Fix random seed")
        set_random_seed(32)

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
    
    # Initialize the PyTorch model
    model = MultitaskRNN(
        input_size=args.input_dim,
        hidden_size_1=args.n_hidden_1,
        hidden_size_2=args.n_hidden_2,
        output_size=args.n_classes,
        dropout=args.p_dropout
    )
    model = model.to(device)

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.learning_rate
    )

    print("Training info:\n")
    print("- Train data: {} samples".format(len(train_dataset)))
    print("- Test data: {} samples".format(len(test_dataset)))
    print("- Batch size: {}".format(args.batch_size))
    print("- Number of epochs: {}".format(args.epochs))
    print("- Learning rate: {}".format(args.learning_rate))
    print("Model config:\n", model)
    trainer = MultitaskOrthogonalTrainer(
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
        output_dir=args.output_dir,
        log_steps=args.log_steps,
        log_wandb=args.log_wandb,
        project_name=args.project_name,
        experiment_name=args.experiment_name,
        weight_regression=args.w_regression,
        weight_classify=args.w_classify,
        weight_grad=args.w_grad,
        weight_trace_norm=args.w_trace_norm

    )
    trainer.train()