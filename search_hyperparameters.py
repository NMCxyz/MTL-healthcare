import os

list_batch_size = [64, 128, 256, 512, 1024]
list_learning_rate = [10e-4, 5 * 10e-4, 10e-3, 5 * 10e-3]
list_hidden_1 = [64, 128, 256, 512, 1024]
list_hidden_2 = [64, 128, 256, 512, 1024]



if __name__ == "__main__":



    os.system("""\
python train_multitask_orthogonal.py \
    --batch_size 256 \
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
    --w_grad 0.33 \
    --data_path "data/train_validate.npz" \
    --output_dir "models/multitask_LSTM" \
    --project_name "Multitask healthcare" \
    --experiment_name "mtl-LSTM-128-64-orthogonal" \
""")

    pass