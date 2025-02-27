import argparse
def handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_name", type=str, required=True,help="The file at this path should contain data in a specific format,For example: {'pred': [list of predictions], 'choices': [list of available choices], 'answers': [list of correct answers]}")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--benchmark_config", type=str,required=True,help="Specify the benchmark. You should provide a value in the format like '--benchmark_config tasks/General/LongBench'.")
    args = parser.parse_args()
    return args
