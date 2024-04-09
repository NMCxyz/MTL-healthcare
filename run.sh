#!/bin/bash

# Define parameters for each instance of the script
params=(
    "--batch_size 4 --epochs 1 --input_dim 3 --n_hidden_1 128 --n_hidden_2 64 --n_classes 12 --p_dropout 0.25 --learning_rate 0.001 --fix_random False --log_steps 5 --w_regression 0.33 --w_classify 0.33 --w_grad 0.33 --data_path 'data/multitask_cls12_regr.npz' --project_name 'Multitask healthcare'"
    "--batch_size 2 --epochs 1 --input_dim 3 --n_hidden_1 128 --n_hidden_2 64 --n_classes 12 --p_dropout 0.25 --learning_rate 0.001 --fix_random False --log_steps 5 --w_regression 0.33 --w_classify 0.33 --w_grad 0.33 --data_path 'data/multitask_cls12_regr.npz' --project_name 'Multitask healthcare'"
    # Add more parameter combinations here as needed
)

# Function to run a single instance of the script
run_instance() {
    python train_multitask_orthogonal.py "$@" --experiment_name "$experiment_name" --output_dir "$output_dir"
}

# Run all instances in parallel
for param_set in "${params[@]}"; do
    # Extract values from parameter set
    n_hidden_1=$(echo "$param_set" | grep -oP '(?<=n_hidden_1 )\d+')
    n_hidden_2=$(echo "$param_set" | grep -oP '(?<=n_hidden_2 )\d+')
    p_dropout=$(echo "$param_set" | grep -oP '(?<=p_dropout )[^ ]+')
    learning_rate=$(echo "$param_set" | grep -oP '(?<=learning_rate )[^ ]+')
    w_classify=$(echo "$param_set" | grep -oP '(?<=w_classify )[^ ]+')
    
    # Create experiment name
    experiment_name="LSTM_n_hidden1-${n_hidden_1}_n_hidden2-${n_hidden_2}_p_dropout-${p_dropout}_${learning_rate}-learning_rate_${w_classify}-weight_classify"
    
    # Create output directory
    output_dir="models/$experiment_name"
    mkdir -p "$output_dir"
    
    # Run the instance with the experiment name and output directory
    run_instance $param_set &
done

# Wait for all instances to finish
wait
