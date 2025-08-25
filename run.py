import os

mode = "filter_design"
n_bands = [3, 6, 11]

seeds = list(range(1))

for seed in seeds:
    for n_band in n_bands:
        os.system(
            f"python deep_learning.py --mode {mode} --learned-bands {n_band} --epochs 3 --seed {seed}"
        )