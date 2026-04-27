
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

    def _on_step(self) -> bool:
        return True
    

    def _on_rollout_end(self) -> None:
        n_days = sum(calculator.data.n_days for calculator in self.test_calculators)
        ic_test_mean = 0.

        test_ics = []
        for i, test_calculator in enumerate(self.test_calculators, start=1):
            ic_test = self.alpha.test(test_calculator)      ## TODO
            ic_test_mean += ic_test * test_calculator.data.n_days / n_days
            self.logger.record(f'test/ic_{i}', ic_test)
            
            test_ics.append(ic_test)
        self.logger.record(f'test/ic_mean', ic_test_mean)
        self.save_checkpoint()

        # --- NEW CLEAN INTERPRETABLE LOGGING ---
        print(f"\n" + "="*50)
        print(f"📊 ROLLOUT COMPLETE | Step: {self.num_timesteps}")
        print(f"📈 Latest Formula Evaluated:")
        print(f"   {self.alpha.expr}")
        print(f"\n🧪 OUT-OF-SAMPLE TEST METRICS (Unseen Regimes):")
        for i, ic in enumerate(test_ics, start=1):
            print(f"   Segment {i}: IC = {ic:.5f}")
        print(f"   -------------------")
        print(f"   Weighted Mean IC: {ic_test_mean:.5f}")
        print("="*50 + "\n")
        
    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_expr.txt', 'w') as f:
            f.write(str(self.alpha.expr) if self.alpha.expr else "None")

    @property
    def alpha(self) -> Alpha:                                          ## change this to get the alpha object from the environment
        assert(isinstance(self.env_core.alpha, Alpha))                
        return self.env_core.alpha

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore



import argparse


def run_single_experiment(
    seed: int = 0,
    instruments: str = "csi300",
    steps: int = 100_000,
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
    save_path = os.path.join("./out/results", name_prefix)
    os.makedirs(save_path, exist_ok=True)

    device = torch.device("cuda:0")                           # TODO: dataparallel on > 1 GPU, kaggle we get 2 GPUs
    
    close = Feature(FeatureType.CLOSE)                        # instantiate Feature object close and set self._feature to FeatureType.CLOSE
    
    target = Ref(close, -20) / close - 1                      # instantiate Ref object and set operand to close and delta_time to -20
                                                              # / instantiates object of Div and sets lhs and rhs
                                                              # - instantiates object of Sub and sets lhs and rhs 

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
        for j in range(i, len(calculators), num_train_clusters):
            test_calculators.append(calculators[j])


        checkpoint_callback = CustomCallback(
            save_path=save_path,
            test_calculators=test_calculators,
            verbose=1
        )
        model = MaskablePPO(
            "MlpPolicy",
            env,
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
            tb_log_name=name_prefix,
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
    parser.add_argument("--steps", type=Optional[int], default=100_000)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse()
    main(args)
