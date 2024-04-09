# # Training multitask
# python train_multitask.py \
#     --batch_size 256 \
#     --epochs 600 \
#     --input_dim 3 \
#     --n_hidden_1 128 \
#     --n_hidden_2 64 \
#     --n_classes 12 \
#     --p_dropout 0.25 \
#     --learning_rate 0.001 \
#     --fix_random False \
#     --log_steps 5 \
#     --data_path "/mnt/new2/acadimic/mtl-healthcare/data/respiration_regression_and_12posture-classification.npz" \
#     --output_dir "models/multitask_LSTM_base" \
#     --project_name="Multitask healthcare" \
#     --experiment_name="mtl-LSTM-128-64" \
#     --log_wandb


# Training multitask orthogonal
python train_multitask_orthogonal.py \
    --batch_size 1024 \
    --epochs 600 \
    --input_dim 3 \
    --n_hidden_1 128 \
    --n_hidden_2 64 \
    --n_classes 12 \
    --p_dropout 0.25 \
    --learning_rate 0.001 \
    --fix_random False \
    --log_steps 5 \
    --w_regression 0.33 \
    --w_classify 0.33 \
    --w_trace_norm 0.001 \
    --w_grad 0.33 \
    --data_path "data/train_validate.npz" \
    --output_dir "models/multitask_LSTM" \
    --project_name "Multitask healthcare" \
    --experiment_name "mtl-LSTM-128-64-orthogonal" \
    # --log_wandb
