import argparse
from fingerprint.pipeline import Pipeline

def main(args):
    # Create a pipeline instance
    pipeline = Pipeline(args)

    # Run the pipeline
    pipeline.build_and_run()

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Pipeline to run multiple commands sequentially with parameters.")

    parser.add_argument('mode', choices=['fingerprint', 'test', 'user', 'erase', 'eval'], help="Mode to run")
    parser.add_argument('--method', choices=['ut', 'all_vocab', 'if_adapter', 'dialogue'], help="Fingerprinting method")
    parser.add_argument('--num_gpus', type=int, default=4, required=False, help='Number of GPUs to use')
    parser.add_argument('--master_port', type=int, default=25000, required=False, help='deepspeed master port')
    parser.add_argument('--config_file', type=str, required=False, default="config/train_config.json", help='Path to the config file')
    
    parser.add_argument('--model_path', type=str, required=False, help='Name of the base model')
    
    parser.add_argument('--multi_fingerprint', action="store_true", help="Use multiple fingerprints. Otherwise use a single fingerprint. Deprecated now.")
    # parser.add_argument('--use_all_vocab', action="store_true", help="Use all vocab. Otherwise use only the under-trained tokens.")
    parser.add_argument('--num_fingerprint', type=int, default=32, required=False, help='Number of fingerprints in dataset. Repeat fingerprints if single fingerprint.')
    parser.add_argument('--num_regularization', type=int, default=128, required=False, help='Number of regularizations in dataset')

    parser.add_argument('--x_length_min', type=int, default=11, required=False, help='Minimum length of x')
    parser.add_argument('--x_length_max', type=int, default=15, required=False, help='Maximum length of x')
    parser.add_argument('--y_length', type=int, default=5, required=False, help='Length of y')

    parser.add_argument('--lr', type=float, default=2e-5, required=False, help='learning rate')
    parser.add_argument('--epoch', type=int, default=30, required=False, help='epochs for training')
    parser.add_argument('--total_bsz', type=int, default=64, required=False, help='total_bsz')

    parser.add_argument('--embedding_only', action="store_true", help="Freeze non-embedding parameters.")
    parser.add_argument('--do_eval', action="store_true", help="Run evaluation after training")
    parser.add_argument('--no_test', action="store_true", help="Do not run the fingerprint test after fingerprinting")

    # for eval
    parser.add_argument('--tasks', nargs='+', required=False, help='List of tasks')
    parser.add_argument('--shots', nargs='+', type=int, required=False, help='List of shots (0, 1, 5)', choices=[0, 1, 5])

    # downstream user
    parser.add_argument("--user_task", type=str, help="user downstream tasks", default=None, choices=["alpaca", "alpaca_gpt4", "dolly", "sharegpt", "ni"])

    # fingerprint test
    parser.add_argument("--num_guess", type=int, default=500, required=False, help="number of fingerprint guesses")
    parser.add_argument("--info_path", type=str, required=False, help="path to the dataset info file")

    args = parser.parse_args()
    main(args)

