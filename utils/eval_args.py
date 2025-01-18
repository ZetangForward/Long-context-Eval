import argparse
def handle_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_path", type=str, required=True,help="The file at this path should contain data in a specific format,For example: {'pred': [list of predictions], 'choices': [list of available choices], 'answers': [list of correct answers]}")
    parser.add_argument("--output_path", type=str, required=True,help="to store the generated evaluation results")
    args = parser.parse_args()
    return args
