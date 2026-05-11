from __future__ import annotations

from typing import Dict

from diffusion_policy.policy.diffusion_unet_timm_mod1_policy import DiffusionUnetTimmMod1Policy


class DiffusionUnetTimmIsaacLabPolicy(DiffusionUnetTimmMod1Policy):
    """Thin adapter to use the UMiFT timm policy with IsaacLab's wrist+state dataset format."""

    def predict_action(self, obs: Dict, debug_action: Dict = None) -> Dict:
        # Current IsaacLab diffusion code passes {"wrist": ..., "state": ...}.
        if "sparse" not in obs:
            obs = {"sparse": obs}
        result = super().predict_action(obs=obs, debug_action=debug_action)
        sparse = result["sparse"]
        # Match the shape/keys used by assess_diffusion.py and the existing DP workspace.
        return {
            "action": sparse,
            "action_pred": sparse,
            "sparse": sparse,
            "dense": result.get("dense", None),
        }

    def compute_loss(self, batch, args=None):
        if args is None:
            args = {
                "start_training_dense": False,
                "dense_traj_cond_use_gt": False,
                "normalization_weight": 0.0,
            }

        # Current IsaacLab dataset returns {"obs": {"wrist","state"}, "action": ...}.
        if "sparse" not in batch["obs"]:
            batch = {
                "obs": {"sparse": batch["obs"]},
                "action": {"sparse": batch["action"]},
            }
        if "dense" in batch.get("obs", {}):
            batch["obs"]["dense"] = batch["obs"]["dense"]
        if "dense" in batch.get("action", {}):
            batch["action"]["dense"] = batch["action"]["dense"]
        return super().compute_loss(batch=batch, args=args)
