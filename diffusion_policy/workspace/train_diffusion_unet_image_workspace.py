if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import time
import re
import dill
import threading
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(float(seconds))), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _get_peak_vram_mib(device: torch.device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        self.resume_training(cfg)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device=device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        run_start = time.time()
        session_start_epoch = self.epoch
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                epoch_start = time.time()
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        if (
                            cfg.training.vram_log_every is not None
                            and cfg.training.vram_log_every > 0
                            and ((self.global_step + 1) % cfg.training.vram_log_every == 0)
                        ):
                            peak_vram_mib = _get_peak_vram_mib(device)
                            if peak_vram_mib is not None:
                                step_log['train_peak_vram_mib'] = peak_vram_mib
                                torch.cuda.reset_peak_memory_stats(device=device)

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                step_log['epoch_time'] = time.time() - epoch_start
                step_log['run_elapsed'] = time.time() - run_start
                completed_session_epochs = max((self.epoch - session_start_epoch + 1), 1)
                avg_epoch_time = step_log['run_elapsed'] / completed_session_epochs
                remaining_epochs = max(cfg.training.num_epochs - (self.epoch + 1), 0)
                step_log['eta_seconds'] = avg_epoch_time * remaining_epochs
                step_log['eta_hms'] = _format_duration(step_log['eta_seconds'])
                peak_vram_mib = _get_peak_vram_mib(device)
                if peak_vram_mib is not None:
                    step_log.setdefault('train_peak_vram_mib', peak_vram_mib)
                    torch.cuda.reset_peak_memory_stats(device=device)

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(epoch=self.epoch)
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = None
                    monitor_key = topk_manager.monitor_key
                    if monitor_key in metric_dict:
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    else:
                        print(
                            f"[WARN] Skipping top-k checkpoint for epoch {self.epoch + 1:03d}: "
                            f"missing monitor key '{monitor_key}'."
                        )

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path, epoch=self.epoch)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                val_loss_text = ""
                if 'val_loss' in step_log:
                    val_loss_text = f"val_loss={step_log['val_loss']:.6f} "
                print(
                    f"[INFO] Epoch {self.epoch + 1:03d}: "
                    f"train_loss={step_log['train_loss']:.6f} "
                    f"{val_loss_text}"
                    f"lr={step_log['lr']:.6e} "
                    f"epoch_time={_format_duration(step_log['epoch_time'])} "
                    f"run_elapsed={_format_duration(step_log['run_elapsed'])} "
                    f"eta={step_log['eta_hms']}"
                )
                self.global_step += 1
                self.epoch += 1

    def save_checkpoint(self, path=None, tag='latest', epoch=0,
            exclude_keys=None,
            include_keys=None,
            use_thread=True):
        should_prune_latest = path is None
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath(
                'checkpoints', f'{tag}_epoch{epoch}.ckpt')
        else:
            path = pathlib.Path(path)
            path = path.with_name(f'{path.stem}_epoch{epoch}{path.suffix}')
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        }

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda: torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
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
        checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
        checkpoint_files = list(checkpoint_dir.glob('latest_epoch*.ckpt'))
        epoch_files = list()
        for file in checkpoint_files:
            match = re.search(r'_epoch(\d+)\.ckpt$', file.name)
            if match:
                epoch_files.append((int(match.group(1)), file))
        if not epoch_files:
            return None
        epoch_files.sort(key=lambda x: x[0], reverse=True)
        return epoch_files[0][1]

    def get_previous_checkpoint_path(self, current_path):
        current_path = pathlib.Path(current_path)
        checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
        checkpoint_files = list(checkpoint_dir.glob('latest_epoch*.ckpt'))
        epoch_files = list()
        for file in checkpoint_files:
            match = re.search(r'_epoch(\d+)\.ckpt$', file.name)
            if match:
                epoch_files.append((int(match.group(1)), file))
        epoch_files.sort(key=lambda x: x[0], reverse=True)
        current_match = re.search(r'_epoch(\d+)\.ckpt$', current_path.name)
        if current_match is None:
            return None
        current_epoch = int(current_match.group(1))
        for epoch, file in epoch_files:
            if epoch < current_epoch:
                return file
        return None


def _prune_checkpoints(checkpoint_dir, keep_last=3):
    checkpoint_files = list(pathlib.Path(checkpoint_dir).glob('latest_epoch*.ckpt'))
    epoch_files = list()
    for file in checkpoint_files:
        match = re.search(r'_epoch(\d+)\.ckpt$', file.name)
        if match:
            epoch_files.append((int(match.group(1)), file))
    epoch_files.sort(key=lambda x: x[0], reverse=True)
    for _, file in epoch_files[keep_last:]:
        try:
            file.unlink()
        except FileNotFoundError:
            pass


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
