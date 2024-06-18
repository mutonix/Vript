import argparse
import pandas as pd
import sys
import os
sys.path.append('vript_bench_verification')
from bench_utils import HALScorer, RRScorer, EROScorer

def main(args):
    if args.task == 'HAL':
        scorer = HALScorer(traditional=False, cache_dir=args.data_cache_dir)
        print("*"*10, "Vript-HAL Task", "*"*10)
    elif args.task == 'RR':
        scorer = RRScorer(task_type='clip', cache_dir=args.data_cache_dir)
        print("*"*10, "Vript-RR Task", "*"*10)
    else:
        scorer = EROScorer(cache_dir=args.data_cache_dir)
        print("*"*10, "Vript-ERO Task", "*"*10)
        
    if args.prediction_file is not None:
        if not os.path.exists(os.path.join(args.prediction_file.split('/')[0], "scores_"+args.prediction_file.split('/')[-1])) and not args.prediction_file.split('/')[-1].startswith("scores_"):
            predictions = pd.read_csv(args.prediction_file)
            print(f"Computing scores of {args.task} for [[{args.prediction_file.split('/')[-1]}]]")
            scorer.compute_scores(predictions, output_file=os.path.join(args.prediction_file.split('/')[0], "scores_"+args.prediction_file.split('/')[-1]))
            print(f"Verification result is saved in {os.path.join(args.prediction_file.split('/')[0], 'scores_'+args.prediction_file.split('/')[-1])}")
    elif args.prediction_dir is not None:
        for file in os.listdir(args.prediction_dir):
            if 'open' in file.lower() and 'rr' in file.lower():
                continue
            
            if file.endswith('.csv') and not os.path.exists(os.path.join(args.prediction_dir, "scores_"+file)) and not file.startswith("scores_"):
                predictions = pd.read_csv(os.path.join(args.prediction_dir, file))
                print(f"Computing scores of {args.task} for [[{file}]]")
                scorer.compute_scores(predictions, output_file=os.path.join(args.prediction_dir, "scores_"+file))
                print(f"Verification result is saved in {os.path.join(args.prediction_dir, 'scores_'+file)}")
                print("--------------------\n")
    else:
        raise ValueError("Please provide either prediction file or prediction directory.")


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction_file', type=str, default=None, help='result file')
    parser.add_argument("--prediction_dir", type=str, default=None, help='result directory')
    parser.add_argument('--task', choices=['HAL', 'RR', 'ERO'], default='RR', help='task')
    parser.add_argument('--data_cache_dir', type=str, default=None, help='data cache directory')
    args = parser.parse_args()
    
    assert args.prediction_file is not None or args.prediction_dir is not None, "Please provide either prediction file or prediction directory."
    main(args)
