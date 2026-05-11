from __future__ import annotations

import copy
import os
import pathlib
import random
import re
import threading

import hydra
import dill
import numpy as np
import torch
import tqdm
import wandb
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.diffusion_unet_timm_isaaclab_policy import DiffusionUnetTimmIsaacLabPolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


class TrainTimmDpWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.model: DiffusionUnetTimmIsaacLabPolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetTimmIsaacLabPolicy | None = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        self.resume_training(cfg)

        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        sparse_normalizer = dataset.get_normalizer()

        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(sparse_normalizer=sparse_normalizer, dense_normalizer=None)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(sparse_normalizer=sparse_normalizer, dense_normalizer=None)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        ema: EMAModel | None = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        env_runner: BaseImageRunner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)

        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging,
        )
        wandb.config.update({"output_dir": self.output_dir})

        topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk)

        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        train_sampling_batch = None
        train_flags = {
            "start_training_dense": False,
            "dense_traj_cond_use_gt": False,
            "normalization_weight": 0.0,
        }

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):
                step_log = {}
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = []
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        raw_loss = self.model.compute_loss(batch, train_flags)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        if cfg.training.use_ema:
                            ema.step(self.model)

                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if cfg.training.max_train_steps is not None and batch_idx >= (cfg.training.max_train_steps - 1):
                            break

                step_log["train_loss"] = float(np.mean(train_losses))

                policy = self.ema_model if cfg.training.use_ema else self.model
                policy.eval()

                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    step_log.update(runner_log)

                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = []
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                val_losses.append(self.model.compute_loss(batch, train_flags))
                                if cfg.training.max_val_steps is not None and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            step_log["val_loss"] = torch.mean(torch.tensor(val_losses)).item()

                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        result = policy.predict_action(batch["obs"])
                        pred_action = result["action_pred"]
                        gt_action = batch["action"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()

                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    metric_dict = {k.replace("/", "_"): v for k, v in step_log.items()}
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                policy.train()
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

    def save_checkpoint(
        self,
        path=None,
        tag="latest",
        epoch=0,
        exclude_keys=None,
        include_keys=None,
        use_thread=True,
    ):
        should_prune_latest = path is None
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath(
                "checkpoints", f"{tag}_epoch{epoch}.ckpt"
            )
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ("_output_dir",)

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"cfg": self.cfg, "state_dicts": dict(), "pickles": dict()}

        for key, value in self.__dict__.items():
            if hasattr(value, "state_dict") and hasattr(value, "load_state_dict"):
                if key not in exclude_keys:
                    if use_thread:
                        payload["state_dicts"][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload["state_dicts"][key] = value.state_dict()
            elif key in include_keys:
                payload["pickles"][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open("wb"), pickle_module=dill)
            )
            self._saving_thread.start()
        else:
            torch.save(payload, path.open("wb"), pickle_module=dill)
        if should_prune_latest:
            _prune_checkpoints(path.parent)
        return str(path.absolute())

    def resume_training(self, cfg):
        if not cfg.training.resume:
            return

        print("Resuming training...")
        latest_ckpt_path = self.get_latest_checkpoint_path()
        if latest_ckpt_path is None:
            fallback_path = self.get_checkpoint_path()
            if fallback_path.is_file():
                latest_ckpt_path = fallback_path

        if latest_ckpt_path is None:
            print("No checkpoints found. Starting from scratch.")
            return

        print(f"Attempting to resume from checkpoint {latest_ckpt_path}")
        while latest_ckpt_path is not None:
            try:
                self.load_checkpoint(path=latest_ckpt_path)
                print(f"Successfully resumed from checkpoint {latest_ckpt_path}")
                return
            except Exception as e:
                print(f"Failed to load checkpoint {latest_ckpt_path}: {e}")
                latest_ckpt_path = self.get_previous_checkpoint_path(latest_ckpt_path)

        print("No valid checkpoints found. Starting from scratch.")

    def get_latest_checkpoint_path(self):
        checkpoint_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
        checkpoint_files = list(checkpoint_dir.glob("latest_epoch*.ckpt"))
        epoch_files = list()
        for file in checkpoint_files:
            match = re.search(r"_epoch(\d+)\.ckpt$", file.name)
            if match:
                epoch_files.append((int(match.group(1)), file))
        if not epoch_files:
            return None
        epoch_files.sort(key=lambda x: x[0], reverse=True)
        return epoch_files[0][1]

    def get_previous_checkpoint_path(self, current_path):
        current_path = pathlib.Path(current_path)
        checkpoint_dir = pathlib.Path(self.output_dir).joinpath("checkpoints")
        checkpoint_files = list(checkpoint_dir.glob("latest_epoch*.ckpt"))
        epoch_files = list()
        for file in checkpoint_files:
            match = re.search(r"_epoch(\d+)\.ckpt$", file.name)
            if match:
                epoch_files.append((int(match.group(1)), file))
        epoch_files.sort(key=lambda x: x[0], reverse=True)
        current_match = re.search(r"_epoch(\d+)\.ckpt$", current_path.name)
        if current_match is None:
            return None
        current_epoch = int(current_match.group(1))
        for epoch, file in epoch_files:
            if epoch < current_epoch:
                return file
        return None


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to("cpu")
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)


def _prune_checkpoints(checkpoint_dir, keep_last=3):
    checkpoint_files = list(pathlib.Path(checkpoint_dir).glob("latest_epoch*.ckpt"))
    epoch_files = list()
    for file in checkpoint_files:
        match = re.search(r"_epoch(\d+)\.ckpt$", file.name)
        if match:
            epoch_files.append((int(match.group(1)), file))
    epoch_files.sort(key=lambda x: x[0], reverse=True)
    for _, file in epoch_files[keep_last:]:
        try:
            file.unlink()
        except FileNotFoundError:
            pass
