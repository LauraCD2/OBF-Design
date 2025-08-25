import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from libs.dataloader import prepare_data
from libs.utils.dl import classify
from libs.train import BaselineModel, FilterDesignModel, BandSelectionModel, BinaryBandSelectionModel
from libs.models import BACKBONES
from libs.utils.metrics import print_results

import pandas as pd

def init_parser():
    parser = argparse.ArgumentParser(
        description="Deep learning Spectral Classification"
    )
    parser.add_argument(
        "--save-name", default="none", type=str, help="Path to save specific experiment"
    )
    parser.add_argument(
        "--dataset", type=str, default="closed_automatic", help="Dataset name"
    )
    parser.add_argument(
        "--split",
        type=dict,
        default={"train": 0.05, "test": 0.95},
        help="Dataset split",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for dataset samples"
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="spectralnet",
        help="Classifier name",
        choices=["spectralnet", "cnn", "lstm", "transformer", "spectralformer"],
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="step_lr",
        choices=["none", "step_lr"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=3,
        help="Step size for learning rate scheduler",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.9, help="Gamma for learning rate scheduler"
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        default=0,
        type=int,
        metavar="N",
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--mode",
        default="filter_design",
        type=str,
        choices=["baseline", "filter_design", "band_selection", "binary_band_selection"],
        help="Mode of operation",
    )

    parser.add_argument(
        "--learned-bands",
        type=int,
        default=6,
        help="Number of bands to select (only for band selection mode)",
    )
    return parser


def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Using device:", device)
    return device


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_model_baseline(datasets, model_args, **kwargs):
    model = BaselineModel(*model_args)
    model.fit(datasets[0], datasets[1])
    return model

def train_model_filter_design(datasets, model_args, save_filters=False, seed=None, **kwargs):
    model = FilterDesignModel(*model_args)

    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"Total parameters: {total_params}")

    model.fit(datasets[0], datasets[1])
    # freeze the filters
    model.freeze_optics()
    # fine-tune the model with the filters
    fine_tune_epochs = 3
    model.epochs = fine_tune_epochs
    model.fit(datasets[0], datasets[1])


    # result = classify(model, datasets[0], datasets[1], save_name=None)
    # dict_metrics, _ = result
    # final_r2 = dict_metrics["test"]["R2"]

    # if final_r2 > 0.90:
    #     mu, sigma = model.model.filter_params()
    #     filters_params = np.stack([mu, sigma], axis=0)
    #     file_path = os.path.join(".", "filters11", f"filter_params_seed_{seed}.npy")
    #     np.save(file_path, filters_params)

    return model

def train_model_band_selection(datasets, model_args, **kwargs):
    """
    Band selection model training

    The band selection learns a subset of bands that are most relevant for the classification task.
    This is modeled as a binary selection problem, where each band is either selected or not selected.

    The training consists of two steps:
    1. Jointly train the network and the band selection layer
    2. Fine-tune the network with the selected bands
    """

    model = BandSelectionModel(*model_args)
    model.set_binarize(False) #  Train the band selection layer
    model.fit(datasets[0], datasets[1])
    model.set_binarize(True)  # Fine-tune the network with the selected bands
    fine_tune_epochs = 10
    model.epochs = fine_tune_epochs
    model.fit(datasets[0], datasets[1])
    mask = model.model.get_binary_mask()
    print(f"Selected bands: {mask.sum()}")
    return model

def train_model_binary_band_selection(datasets, model_args, **kwargs):

    model = BinaryBandSelectionModel(*model_args)
    model.fit(datasets[0], datasets[1])
    mask = model.model.get_binary_mask()
    print(f"Selected bands: {mask.sum()}")
    return model

def train_model(datasets, model_args, mode, save_filters, seed):
    train_funcs = {
        "baseline": train_model_baseline,
        "filter_design": train_model_filter_design,
        "band_selection": train_model_band_selection,
        "binary_band_selection": train_model_binary_band_selection,
    }
    if mode in train_funcs:
        return train_funcs[mode](datasets, model_args, save_filters=save_filters, seed=seed)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented")


def save_results(save_path, outputs):
    # outputs = [ y_train, y_pred_train, y_test, y_pred_test]
    os.makedirs(save_path, exist_ok=True)
    filenames = ["y_train", "y_pred_train", "y_test", "y_pred_test"]
    for i, filename in enumerate(filenames):
        np.save(os.path.join(save_path, f"{filename}.npy"), outputs[i])

def main():
    parser = init_parser()
    args = parser.parse_args()

    device = set_device()
    set_seed(args.seed)

    dataset_params = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_dataset, test_dataset, num_bands, _ = prepare_data(
        args.dataset,
        args.split,
        args.seed,
        dl=True,
        dataset_params=dataset_params,
        device=device,
    )

    # plot size of dataset
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    backbone, net_config = BACKBONES[args.classifier]
    net_config["input_dim"] = num_bands
    net_config["num_classes"] = 1

    params = dict(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
    )

    datasets   = (train_dataset, test_dataset)

    if args.mode == "baseline":        
        model_args = (backbone, net_config, params, device)
    else:
        model_args = (backbone, net_config, params, args.learned_bands, device)

    save_filters=False
    seed = args.seed
    regressor = train_model(datasets, model_args, args.mode, save_filters, seed)



    result1 = classify(regressor, train_dataset, test_dataset, save_name=args.save_name)
    dict_metrics, outputs = result1
    print_results(args.classifier, args.dataset, dict_metrics)

    columns = ["method", "n_bands", "mape", "mae", "r2"]
    final_mape = dict_metrics["test"]["MAPE"]
    final_mae = dict_metrics["test"]["MAE"]
    final_r2 = dict_metrics["test"]["R2"]
    method = args.mode

    n_bands = args.learned_bands if args.mode != "baseline" else num_bands

    # check if file exists, if not create it
    # if file exists, append to it
    results_name = "results.csv"
    if not os.path.exists(results_name):
        df = pd.DataFrame(columns=columns)
        df.to_csv(results_name, index=False)

    df = pd.read_csv(results_name)

    # append new results to the file
    new_row = pd.DataFrame([[method, n_bands, final_mape, final_mae, final_r2]], columns=columns)
    df = pd.concat([df, new_row], ignore_index=True)
    # save the updated file
    df.to_csv(results_name, index=False)

    outputs = [out.cpu().numpy() for out in outputs]

    save_path = os.path.join(".", "results", args.classifier)
    save_results(
        save_path,
        outputs
    )


if __name__ == "__main__":
    main()
