import time
from typing import Dict

import torch

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


def _to_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _center_crop_torch(images: torch.Tensor, crop_size: int | None) -> torch.Tensor:
    if crop_size is None:
        return images
    if crop_size <= 0:
        raise ValueError(f"image_crop_size must be positive when set, got {crop_size}")
    height, width = images.shape[1:3]
    if crop_size > height or crop_size > width:
        raise ValueError(
            f"image_crop_size={crop_size} exceeds image dimensions {(height, width)}."
        )
    top = (height - crop_size) // 2
    left = (width - crop_size) // 2
    return images[:, top : top + crop_size, left : left + crop_size, :]


def _get_current_success_rate(env):
    if not hasattr(env, "_get_curr_successes"):
        return None
    check_rot = getattr(env.cfg_task, "name", None) == "nut_thread"
    curr_successes = env._get_curr_successes(
        success_threshold=env.cfg_task.success_threshold,
        check_rot=check_rot,
    )
    return torch.count_nonzero(curr_successes).float() / env.num_envs


def _get_episode_success_rate(env):
    if not hasattr(env, "ep_succeeded"):
        return None
    return torch.count_nonzero(env.ep_succeeded).float() / env.num_envs


def _resolve_ft_body_indices(env):
    robot = env.scene["robot"]
    left_ids, _ = robot.find_bodies("fr3_left_ft")
    right_ids, _ = robot.find_bodies("fr3_right_ft")
    if len(left_ids) == 0 or len(right_ids) == 0:
        raise ValueError("Could not resolve FT bodies 'fr3_left_ft' and 'fr3_right_ft'.")
    return robot, int(left_ids[0]), int(right_ids[0])


class IsaacLabDiffusionImageRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        task_name: str,
        num_envs: int = 8,
        num_loops: int = 1,
        device: str = "cuda:0",
        use_fabric: bool = True,
        headless: bool = True,
        real_time: bool = False,
        random_orn: float | None = None,
        image_crop_size: int | None = None,
        use_ft: bool = False,
    ):
        super().__init__(output_dir)
        self.task_name = task_name
        self.num_envs = int(num_envs)
        self.num_loops = int(num_loops)
        self.device = device
        self.use_fabric = bool(use_fabric)
        self.headless = bool(headless)
        self.real_time = bool(real_time)
        self.random_orn = random_orn
        self.image_crop_size = image_crop_size
        self.use_ft = bool(use_ft)

        self._app_launcher = None
        self._simulation_app = None
        self._env = None
        self._base_env = None
        self._gym = None

    def _ensure_env(self):
        if self._env is not None:
            return self._env, self._base_env

        from isaaclab.app import AppLauncher

        app_cfg = {
            "headless": self.headless,
            "enable_cameras": True,
            "device": self.device,
        }
        self._app_launcher = AppLauncher(app_cfg)
        self._simulation_app = self._app_launcher.app

        import gymnasium as gym
        from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
        from isaaclab_tasks.utils import parse_env_cfg

        import isaaclab_tasks  # noqa: F401
        import fr3_manipulation.tasks  # noqa: F401

        env_cfg = parse_env_cfg(
            self.task_name,
            device=self.device,
            num_envs=self.num_envs,
            use_fabric=self.use_fabric,
        )
        env_cfg.scene.clone_in_fabric = False
        if hasattr(env_cfg, "seed") and env_cfg.seed is None:
            env_cfg.seed = 42
        if self.random_orn is not None and hasattr(env_cfg, "task") and hasattr(env_cfg.task, "randomize_hand_init_tilt"):
            env_cfg.task.randomize_hand_init_tilt = True
            env_cfg.task.hand_init_tilt_noise_deg = self.random_orn

        env = gym.make(self.task_name, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        self._gym = gym
        self._env = env
        self._base_env = env.unwrapped
        return self._env, self._base_env

    def _build_current_obs(self, env):
        if getattr(env, "_wrist_camera", None) is None:
            raise RuntimeError(
                "Isaac Lab rollout requires wrist camera data, but the environment has no wrist camera."
            )

        wrist = env._wrist_camera.data.output["rgb"][..., :3]
        wrist = _center_crop_torch(wrist, self.image_crop_size)
        wrist = wrist.permute(0, 3, 1, 2).contiguous().float() / 255.0
        gripper_pos = torch.mean(env.joint_pos[:, 7:], dim=1, keepdim=True)
        state_parts = [env.fingertip_midpoint_pos, env.fingertip_midpoint_quat, gripper_pos]
        if self.use_ft:
            robot, left_ft_body_idx, right_ft_body_idx = _resolve_ft_body_indices(env)
            left_ft_wrench = robot.data.body_incoming_joint_wrench_b[:, left_ft_body_idx]
            right_ft_wrench = robot.data.body_incoming_joint_wrench_b[:, right_ft_body_idx]
            state_parts.extend((left_ft_wrench, right_ft_wrench))
        state = torch.cat(state_parts, dim=-1)
        return wrist, state

    @torch.inference_mode()
    def run(self, policy: BaseImagePolicy) -> Dict:
        env, base_env = self._ensure_env()
        env.reset()

        wrist_history = None
        state_history = None
        action_plan = None
        action_step = 0
        completed_loops = 0
        timestep = 0
        episode_success_rates = list()
        final_success_rates = list()
        n_obs_steps = int(policy.n_obs_steps)
        steps_per_loop = max(int(base_env.max_episode_length) - 1, 1)
        max_steps = self.num_loops * steps_per_loop if self.num_loops > 0 else None

        while True:
            loop_start = time.time()
            wrist, state = self._build_current_obs(base_env)
            if wrist_history is None or state_history is None:
                wrist_history = wrist.unsqueeze(1).repeat(1, n_obs_steps, 1, 1, 1)
                state_history = state.unsqueeze(1).repeat(1, n_obs_steps, 1)
            else:
                wrist_history = torch.roll(wrist_history, shifts=-1, dims=1)
                wrist_history[:, -1] = wrist
                state_history = torch.roll(state_history, shifts=-1, dims=1)
                state_history[:, -1] = state

            if action_plan is None or action_step >= action_plan.shape[1]:
                result = policy.predict_action({"wrist": wrist_history, "state": state_history})
                action_plan = result["action"]
                action_step = 0

            action = action_plan[:, action_step]
            action_step += 1
            _, _, terminated, truncated, extras = env.step(action)
            dones = torch.logical_or(terminated, truncated)

            if len(dones) > 0 and torch.all(dones).item():
                completed_loops += 1
                episode_success_rate = _get_episode_success_rate(base_env)
                final_success_rate = extras.get("successes") if isinstance(extras, dict) else None
                if final_success_rate is None:
                    final_success_rate = _get_current_success_rate(base_env)
                if episode_success_rate is not None:
                    episode_success_rates.append(episode_success_rate.detach().cpu())
                if final_success_rate is not None:
                    final_success_rates.append(torch.as_tensor(final_success_rate).detach().cpu())
                wrist_history = None
                state_history = None
                action_plan = None
                action_step = 0
                if self.num_loops > 0 and completed_loops >= self.num_loops:
                    break

            timestep += 1
            if max_steps is not None and timestep >= max_steps:
                break

            sleep_time = float(base_env.step_dt) - (time.time() - loop_start)
            if self.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        metrics = {
            "rollout_num_loops": completed_loops,
            "rollout_num_envs": base_env.num_envs,
        }
        if episode_success_rates:
            mean_episode_success_rate = torch.stack(episode_success_rates).mean()
            metrics["rollout_episode_success_rate"] = _to_float(mean_episode_success_rate)
        if final_success_rates:
            mean_final_success_rate = torch.stack(final_success_rates).float().mean()
            metrics["rollout_final_success_rate"] = _to_float(mean_final_success_rate)
        return metrics

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
            self._base_env = None
        if self._simulation_app is not None:
            self._simulation_app.close()
            self._simulation_app = None
            self._app_launcher = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
