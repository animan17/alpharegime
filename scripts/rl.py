
import json
import os
from typing import Optional, Tuple, List
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import fire

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from alphagen.data.expression import *
from alphagen.data.parser import ExpressionParser
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils import reseed_everything, get_logger
from alphagen.rl.env.core import AlphaEnvCore
from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.stock_data import initialize_qlib
from alphagen.data.calculator import AlphaCalculator
from alphagen.utils.clustering import kmeans, calc_clusters, plot_cluster_waves
from alphagen.utils.alpha import Alpha


import argparse


def build_parser() -> ExpressionParser:
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        }
    )
    

class CustomCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        test_calculators: List[QLibStockDataCalculator],
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)
        self.best_ic = -float('inf') 

    def _on_step(self) -> bool:
        return True
    

    def _on_rollout_end(self) -> None:
        # --- FIX: Get the true number of pairs in the cluster across all test sets ---
        total_cluster_pairs = sum(len(calculator.days) for calculator in self.test_calculators)
        ic_test_mean = 0.

        test_ics = []
        for i, test_calculator in enumerate(self.test_calculators, start=1):
            ic_test = self.alpha.test(test_calculator)      
            
            # --- FIX: Weight by the actual number of pairs in this specific segment ---
            segment_cluster_pairs = len(test_calculator.days)
            weight = segment_cluster_pairs / total_cluster_pairs if total_cluster_pairs > 0 else 0
            
            ic_test_mean += ic_test * weight
            self.logger.record(f'test/ic_{i}', ic_test)
            
            test_ics.append(ic_test)
        
        self.logger.record(f'test/ic_mean', ic_test_mean)

        # --- NEW: ONLY SAVE IF IT IS THE BEST ---
        is_best = False
        if ic_test_mean > self.best_ic:
            self.best_ic = ic_test_mean
            self.save_checkpoint(is_best=True)
            is_best = True
        else:
            self.save_checkpoint(is_best=False) # Still save latest just in case

        # --- NEW CLEAN INTERPRETABLE LOGGING ---
        print(f"\n" + "="*50)
        print(f"📊 ROLLOUT COMPLETE | Step: {self.num_timesteps}")
        if is_best:
            print(f"⭐ NEW BEST MODEL SAVED! (IC: {self.best_ic:.5f})")
        print(f"📈 Latest Formula Evaluated:")
        print(f"   {self.alpha.expr}")
        print(f"\n🧪 OUT-OF-SAMPLE TEST METRICS (Unseen Regimes):")
        for i, ic in enumerate(test_ics, start=1):
            print(f"   Segment {i}: IC = {ic:.5f}")
        print(f"   -------------------")
        print(f"   Weighted Mean IC: {ic_test_mean:.5f}")
        print("="*50 + "\n")
        
    def save_checkpoint(self, is_best: bool = False):
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_expr.txt', 'w') as f:
            f.write(str(self.alpha.expr) if self.alpha.expr else "None")

        # Overwrite the 'best' checkpoint
        if is_best:
            best_path = os.path.join(self.save_path, 'best_model')
            self.model.save(best_path)
            with open(f'{best_path}_expr.txt', 'w') as f:
                f.write(str(self.alpha.expr) if self.alpha.expr else "None")

    @property
    def alpha(self) -> Alpha:                                          ## change this to get the alpha object from the environment
        assert(isinstance(self.env_core.alpha, Alpha))                
        return self.env_core.alpha

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore




def evaluate_synergistic_ensemble(
    base_save_path: str,
    test_datasets: List[StockData],
    test_clusters: List[List[Tuple[np.ndarray, np.ndarray]]],
    target: Expression,
    device: torch.device
):
    parser = build_parser()
    close_feat = Feature(FeatureType.CLOSE)
    num_clusters = len(test_clusters[0])
    expressions = []
    
    # 1. Load all 10 agents' best formulas
    print("\n" + "="*50)
    print("🔍 LOADING SYNERGISTIC ENSEMBLE")
    for i in range(num_clusters):
        expr_file = os.path.join(base_save_path, f"cluster_{i}", "best_model_expr.txt")
        if os.path.exists(expr_file):
            with open(expr_file, 'r') as f:
                expr_str = f.read().strip()
            try:
                expr = parser.parse(expr_str)
            except:
                expr = None
        else:
            expr = None
            expr_str = "File Not Found"
            
        expressions.append(expr)
        print(f"   Cluster {i} Formula: {expr_str}")
        
    print("="*50)
    
    # 2. Evaluate Segment by Segment
    segment_ics = []
    segment_weights = []
    
    for seg_idx, dataset in enumerate(test_datasets):
        n_days, n_stocks = dataset.n_days, dataset.n_stocks
        
        # Initialize an empty market map
        global_alpha = torch.full((n_days, n_stocks), torch.nan, device=device)
        global_target = target.evaluate(dataset)  
        dense_close = close_feat.evaluate(dataset) 
        
        total_pairs_in_segment = 0
        
        # Stitch the cluster predictions together into the global map
        for cluster_idx in range(num_clusters):
            expr = expressions[cluster_idx]
            if expr is None:
                continue
                
            days, stocks = test_clusters[seg_idx][cluster_idx]
            if len(days) == 0:
                continue
                
            days_t = torch.tensor(days, dtype=torch.long, device=device)
            stocks_t = torch.tensor(stocks, dtype=torch.long, device=device)
            
            # Evaluate this specific agent's formula
            dense_out = expr.evaluate(dataset)
            
            # Extract only its designated cluster pairs
            sparse_out = dense_out[days_t, stocks_t]
            sparse_close = dense_close[days_t, stocks_t]
            
            # Apply the required stationarity transformation
            transformed_alpha = (sparse_out / sparse_close) - 1
            
            # Insert the predictions directly into the global market tensor
            global_alpha[days_t, stocks_t] = transformed_alpha
            total_pairs_in_segment += len(days)
            
        # 3. Calculate True Global IC for this Time Segment
        valid_mask = torch.isfinite(global_alpha) & torch.isfinite(global_target)
        val1 = global_alpha[valid_mask]
        val2 = global_target[valid_mask]
        
        if len(val1) > 1:
            # Flattened Pearson Correlation
            mean1, mean2 = val1.mean(), val2.mean()
            var1, var2 = val1.var(), val2.var()
            cov = ((val1 - mean1) * (val2 - mean2)).mean()
            ic = cov / torch.sqrt(var1 * var2 + 1e-8)
            ic_val = ic.item()
        else:
            ic_val = 0.0
            
        segment_ics.append(ic_val)
        segment_weights.append(total_pairs_in_segment)
        print(f"🧪 Segment {seg_idx + 1} Synergistic IC: {ic_val:.5f} (Trading Pairs: {total_pairs_in_segment})")
        
    # 4. Calculate Final Weighted IC
    total_pairs = sum(segment_weights)
    if total_pairs > 0:
        final_ic = sum(ic * w for ic, w in zip(segment_ics, segment_weights)) / total_pairs
    else:
        final_ic = 0.0
        
    print("-" * 50)
    print(f"🌟 FINAL OVERALL SYNERGISTIC IC: {final_ic:.5f}")
    print("=" * 50)
    
    return final_ic



def run_single_experiment(
    seed: int = 0,
    instruments: str = "csi300",
    steps: int = 500_000,
):
    reseed_everything(seed)
    initialize_qlib()

    print(f"""[Main] Starting training process
    Seed: {seed}
    Instruments: {instruments}
    Total Iteration Steps: {steps}
    """)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    name_prefix = f"{instruments}_{seed}_{timestamp}"
    save_path = os.path.join("/kaggle/working/results", name_prefix)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda:0")                           # TODO: dataparallel on > 1 GPU, kaggle we get 2 GPUs
    
    close = Feature(FeatureType.CLOSE)                        # instantiate Feature object close and set self._feature to FeatureType.CLOSE
    
    target = Ref(close, -20) / close - 1                      # instantiate Ref object and set operand to close and delta_time to -20
                                                              # / instantiates object of Div and sets lhs and rhs
                                                              # - instantiates object of Sub and sets lhs and rhs 
                                                              # we didnt add 1e-8 or something to the denominator 'close' because we don't want to hide the error (missing close data)

    def get_dataset(start: str, end: str) -> StockData:
        return StockData(
            instrument=instruments,
            start_time=start,
            end_time=end,
            device=device
        )

    segments = [
        ("2012-01-01", "2021-12-31"),         # train
        ("2022-01-01", "2022-06-30"),         # test
        ("2022-07-01", "2022-12-31"),         # test
        ("2023-01-01", "2023-06-30")          # test
    ]

    datasets = [get_dataset(*s) for s in segments]


    barycenters, train_clusters = kmeans(datasets[0], n_clusters=10, lookback=20)           ### TODO: FINETUNE THIS
    plot_cluster_waves(barycenters, train_clusters, datasets[0], lookback=20, num_samples=100)


    test_clusters = []


    for i in range(1, len(datasets)):
        clusters = calc_clusters(barycenters, datasets[i])
        test_clusters.append(clusters)



    num_train_clusters = len(train_clusters)

    calculators = []

    for days, stocks in train_clusters:
        days_tensor = torch.tensor(days, dtype=torch.long, device=device)
        stocks_tensor = torch.tensor(stocks, dtype=torch.long, device=device)
        calculators.append(QLibStockDataCalculator(datasets[0], days_tensor, stocks_tensor, target))

    
    for i, clusters in enumerate(test_clusters):
        for days, stocks in clusters:
            days_tensor = torch.tensor(days, dtype=torch.long, device=device)
            stocks_tensor = torch.tensor(stocks, dtype=torch.long, device=device)
            calculators.append(QLibStockDataCalculator(datasets[1+i], days_tensor, stocks_tensor, target))



    for i in range(num_train_clusters):               
    
        alpha = Alpha(
            calculator=calculators[i],
            device=device
        )


        env = AlphaEnv(
            alpha,
            device=device,
            print_expr=False
        )

        test_calculators = []
        for j in range(i + num_train_clusters, len(calculators), num_train_clusters):
            test_calculators.append(calculators[j])


        
        cluster_save_path = os.path.join(save_path, f"cluster_{i}")
        os.makedirs(cluster_save_path, exist_ok=True)

        checkpoint_callback = CustomCallback(
            save_path=cluster_save_path,
            test_calculators=test_calculators,
            verbose=1
        )
        model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=0.0001,
            policy_kwargs=dict(
                features_extractor_class=LSTMSharedNet,
                features_extractor_kwargs=dict(
                    n_layers=2,
                    d_model=128,
                    dropout=0.1,
                    device=device,
                ),
            ),
            gamma=1.,
            ent_coef=0.01,
            batch_size=128,
            tensorboard_log="./out/tensorboard",
            device=device,
            verbose=1,
        )
        model.learn(
            total_timesteps=steps,
            callback=checkpoint_callback,
            tb_log_name=f"{name_prefix}_cluster_{i}",
        )

        # --- ADD THIS CLEANUP BLOCK EXACTLY HERE ---
        del model
        del env
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        # ------------------------------------------

    
    # The loop is over, all 10 clusters are trained and saved.
    evaluate_synergistic_ensemble(
        base_save_path=save_path,
        test_datasets=datasets[1:],  # Only pass the 3 test segments
        test_clusters=test_clusters,
        target=target,
        device=device
    )


def main(args):
    random_seeds = args.random_seeds
    for s in random_seeds:
        run_single_experiment(
            seed=s,
            instruments=args.instruments,
            steps=args.steps,                              # TODO:Finetune
        )



def parse():
    """
    :param random_seeds: Random seeds
    :param instruments: Stock subset name
    :param steps: Total iteration steps
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seeds", type=int, nargs='+', default=[0])
    parser.add_argument("--instruments", type=str, default="csi300")
    parser.add_argument("--steps", type=int, default=500_000)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    main(args)
