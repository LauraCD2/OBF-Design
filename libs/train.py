from abc import ABC, abstractmethod

import torch

from torch import nn
from tqdm import tqdm

from .models.optics import BandSelection, FilterDesign, BinaryBandSelection
from .models.regularizers import binary_reg, num_band_reg
from .utils.metrics import AverageMeter, mae_fn, mape_fn, r2_fn


class RegressionModel:
    def __init__(
        self,
        backbone: nn.Module,
        net_config: dict,
        params: dict,
        device="cpu",
    ):

        self.lr = params["lr"]
        weight_decay = params["weight_decay"]
        scheduler = params["scheduler"]
        step_size = params["step_size"]
        gamma = params["gamma"]

        self.epochs = params["epochs"]
        self.device = device

        self.model     = self.bulid_backbone(backbone, net_config)
        self.criterion = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=weight_decay
        )

        if scheduler == "step_lr":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        else:
            self.scheduler = None
        
        self.tracked_metrics = ["mae", "mape"]

    @abstractmethod
    def bulid_model(self, backbone, net_config, mode):
        pass

    @abstractmethod
    def compute_loss(self, y_pred, y_true):
        pass

    def fit(self, dataset, test_dataset=None):
        for epoch in range(self.epochs):
            self.model.train()
            self.run_epoch(dataset, epoch, training=True)
            if self.scheduler is not None:
                self.scheduler.step()
            
            if test_dataset is not None:
                self.test(test_dataset)

    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            return self.run_epoch(dataset, training=False)

    def run_epoch(self, dataset, epoch=0, training=True):
        trackers = self.initialize_trackers()
        data_loop = tqdm(enumerate(dataset), total=len(dataset))



        for i, data in data_loop:
            x, y = data
            y_pred, losses, total_loss = self.forward_pass(x, y, training)

            if training:
                total_loss.backward()
                self.optimizer.step()

            if i ==0:
                y_true_list = y
                y_pred_list = y_pred
            else:
                y_true_list = torch.cat([y_true_list, y], dim=0)
                y_pred_list = torch.cat([y_pred_list, y_pred], dim=0)
                
            r2_score = r2_fn(y_true_list, y_pred_list).item()
            self.update_trackers(trackers, losses, y, y_pred, x.size(0))
            self.log_progress(data_loop, epoch, training, trackers, r2_score)

        if not training:
            y_true = y_true_list
            y_pred = y_pred_list
            return trackers["loss"].avg, trackers["mae"].avg, trackers["mape"].avg, r2_score, (y_true, y_pred)

    def forward_pass(self, x, y, training):
        if training:
            self.optimizer.zero_grad()

        y_pred = self.model(x)
        losses = self.compute_loss(y_pred, y)
        total_loss = sum(losses.values())
        return y_pred, losses, total_loss

    def initialize_trackers(self):
        trackers = {metric: AverageMeter(metric) for metric in self.tracked_metrics}
        for loss in self.tracked_losses:
            trackers[loss] = AverageMeter(loss, fmt=":.4e")
        return trackers

    def update_trackers(self, trackers, losses, y, y_pred, batch_size):
        for loss in self.tracked_losses:
            trackers[loss].update(losses[loss].item(), batch_size)

        trackers["mae"].update(mae_fn(y, y_pred).item(), batch_size)
        trackers["mape"].update(mape_fn(y, y_pred).item(), batch_size)

    def log_progress(self, data_loop, epoch, training, trackers, r2_score):
        data_loop.set_description(f"{'Train' if training else 'Test'} Epoch {epoch + 1}/{self.epochs}, lr: {self.lr}")
        metrics_log = {k: f"{v.avg:.4f}" for k, v in trackers.items()}
        metrics_log["R2"] = f"{r2_score:.4f}"
        data_loop.set_postfix(metrics_log)
        data_loop.bar_format = "{l_bar}{bar:10}{r_bar}"
        data_loop.colour = "cyan" if training else "magenta"
        data_loop.set_postfix_str(f"\033[1;{'36' if training else '35'}m{data_loop.postfix}\033[0m")

class BaselineModel(RegressionModel):

    def __init__(self, backbone, net_config, params, device="cpu"):
        super().__init__(backbone, net_config, params, device)
        self.tracked_losses = ["loss"]

    def bulid_backbone(self, backbone, net_config):
        return backbone(**net_config).to(self.device)
    
    def compute_loss(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)
        return {"loss": loss}
    

class BandSelectionModel(RegressionModel):

    def __init__(self, backbone, net_config, params, learned_bands, device="cpu"):
        self.learned_bands = learned_bands
        super().__init__(backbone, net_config, params, device)
        self.regularize = True # Regularize the number of bands and the binary mask
        self.binary_w = 10.0
        self.num_band_w = 0.1
        self.set_binarize(False)

    def bulid_backbone(self, backbone, net_config):
        backbone = backbone(**net_config)
        return BandSelection(net_config["input_dim"], backbone, learned_bands=self.learned_bands).to(self.device)
    
    def compute_loss(self, y_pred, y_true):

        loss = self.criterion(y_pred, y_true)

        if not self.regularize:
            return {"loss": loss}
        
        binary_reg_loss   = binary_reg(self.model.mask) * self.binary_w
        num_band_reg_loss = num_band_reg(self.model.mask, N=self.learned_bands) * self.num_band_w

        return {"loss": loss, "Bin": binary_reg_loss, "NBands": num_band_reg_loss}

    def set_binarize(self, binarize):
        # if binarize is True, the mask is binarized and not regularized
        if binarize:
            self.model.binarize = True
            self.regularize = False
            self.tracked_losses = ["loss"]
        else:
            self.model.binarize = False
            self.regularize = True
            self.tracked_losses = ["loss", "Bin", "NBands"]


class BinaryBandSelectionModel(RegressionModel):

    def __init__(self, backbone, net_config, params, learned_bands, device="cpu"):
        super().__init__(backbone, net_config, params, device)
        self.learned_bands = learned_bands
        self.num_band_w = 300.0
        self.tracked_losses = ["loss", "NBands"]

    def bulid_backbone(self, backbone, net_config):
        backbone = backbone(**net_config)
        return BinaryBandSelection(net_config["input_dim"], backbone).to(self.device)
    
    def compute_loss(self, y_pred, y_true):

        loss = self.criterion(y_pred, y_true)
        mask = self.model.get_binary_mask()
        num_band_reg_loss = num_band_reg(mask, N=self.learned_bands) * self.num_band_w

        return {"loss": loss, "NBands": num_band_reg_loss}       

class FilterDesignModel(RegressionModel):

    def __init__(self, backbone, net_config, params, learned_bands, device="cpu"):
        self.learned_bands = learned_bands
        super().__init__(backbone, net_config, params, device)
        self.tracked_losses = ["loss"]

    def bulid_backbone(self, backbone, net_config):
        n_filters = self.learned_bands  
        n_bands   = net_config["input_dim"]
        net_config["input_dim"] = n_filters
        backbone  = backbone(**net_config)
        return FilterDesign(n_bands, n_filters, backbone).to(self.device)

    def compute_loss(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)
        return {"loss": loss}


    def freeze_optics(self):
        self.model.mu.requires_grad = False
        self.model.sigma.requires_grad = False
