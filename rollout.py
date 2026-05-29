"""3-way rollout video: orig | recon | pred — full trajectory length."""
import os, sys
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
import wandb
wandb.login = lambda **kwargs: None

import hydra
import torch
import logging
import warnings
from einops import rearrange
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rollout_videos import build_trainer, denorm

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


def to_frames(visual):
    return [denorm(visual[t]).permute(1, 2, 0).cpu().numpy() for t in range(visual.shape[0])]


def save_gif(frames, path, fps=8, target_size=224):
    pil = [Image.fromarray(f).resize((target_size, target_size), Image.BILINEAR) for f in frames]
    pal = pil[0].quantize(colors=64, method=Image.MEDIANCUT, dither=Image.FLOYDSTEINBERG)
    out = [pal] + [f.quantize(palette=pal, dither=Image.FLOYDSTEINBERG) for f in pil[1:]]
    dur_ms = int(1000 / max(1, fps))
    out[0].save(path, save_all=True, append_images=out[1:],
                duration=dur_ms, loop=0, optimize=True)


@hydra.main(config_path="conf", version_base="1.1")
def main(cfg):
    orig_cwd = hydra.utils.get_original_cwd()
    output_dir = cfg.get("output_dir", f"rollout_threeway/{cfg.env.name}")
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(orig_cwd, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    traj_idx = int(cfg.get("traj_idx", 0))
    fps_out = int(cfg.get("fps", 8))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frameskip = cfg.frameskip
    num_hist = cfg.num_hist
    num_pred = cfg.num_pred

    trainer = build_trainer(cfg)
    model = trainer.model
    env_name = cfg.env.name
    log.info(f"Output: {output_dir}")

    dset = trainer.val_traj_dset if trainer.val_traj_dset is not None and len(trainer.val_traj_dset) > 0 else trainer.train_traj_dset

    # Multi-task: if task_name is passed, remap traj_idx to be within that task's range
    task_name = cfg.get("task_name", None)
    if task_name is not None and hasattr(dset, "num_tasks") and dset.num_tasks > 1:
        if task_name not in dset.task_names:
            log.error(f"task_name {task_name!r} not in dset.task_names={dset.task_names}"); return
        t = dset.task_names.index(task_name)
        t_start, t_end = dset._cumulative[t], dset._cumulative[t + 1] - 1
        n_avail = t_end - t_start + 1
        traj_idx = t_start + max(0, min(traj_idx, n_avail - 1))
        log.info(f"  multitask: task={task_name} range=[{t_start},{t_end}] using global idx={traj_idx}")
    else:
        traj_idx = max(0, min(traj_idx, len(dset) - 1))
    obs, act, state, _ = dset[traj_idx]

    horizon = (obs["visual"].shape[0] - 1) // frameskip
    for k in obs.keys():
        obs[k] = obs[k][0 : horizon * frameskip + 1 : frameskip]
    act = act[0 : horizon * frameskip]
    act = rearrange(act, "(h f) d -> h (f d)", f=frameskip)

    obs_g = {k: v.unsqueeze(0).to(device) for k, v in obs.items()}
    act_g = act.unsqueeze(0).to(device)
    state_g = state[0 : horizon * frameskip + 1 : frameskip].unsqueeze(0).to(device) if state is not None else None

    T_in = obs_g["visual"].shape[1]
    log.info(f"  trajectory length after frameskip: T_in={T_in} frames")

    with torch.no_grad():
        # === ORIG ===
        gt_visual = obs["visual"]  # (T_in, 3, H, W), [-1, 1]

        # === RECON === encode full traj, decode each frame.
        # obs has T_in frames, act has T_in-1 (frameskip math). Trim obs to act_len.
        T_act = act_g.shape[1]
        obs_trim = {k: v[:, :T_act] for k, v in obs_g.items()}
        z_full, z_dct_full = model.encode(obs_trim, act_g)
        z_proj_full = model.separate_s_a_p(z_full)["projected"]
        dec_out, _ = model.decoder.forward_with_losses(
            z_proj_full,
            targets={
                "visual_emb": z_dct_full["visual"],
                "image": obs_trim["visual"],
                "action": act_g,
            },
        )
        recon_full = rearrange(dec_out, "(b t) c h w -> b t c h w", b=1)
        recon_visual = recon_full[0].cpu()

        # === PRED === open-loop rollout from obs[:num_hist], roll for full action length
        obs_0 = {k: v[:, :num_hist] for k, v in obs_g.items()}
        z_dct_r, z_r = model.rollout(obs_0, act_g)
        if cfg.has_decoder:
            z_emb = model.emb_decoder(z_dct_r["projected"])
            pred_obs, _ = model.decode(z_emb)
            pred_visual = pred_obs["visual"][0].cpu()
        else:
            pred_visual = recon_visual.clone()

        T = min(gt_visual.shape[0], recon_visual.shape[0], pred_visual.shape[0])
        gt_visual = gt_visual[:T]; recon_visual = recon_visual[:T]; pred_visual = pred_visual[:T]
        log.info(f"  T={T}  orig={tuple(gt_visual.shape)} recon={tuple(recon_visual.shape)} pred={tuple(pred_visual.shape)}")

        save_gif(to_frames(gt_visual),    os.path.join(output_dir, f"{env_name}_orig.gif"),  fps=fps_out)
        save_gif(to_frames(recon_visual), os.path.join(output_dir, f"{env_name}_recon.gif"), fps=fps_out)
        save_gif(to_frames(pred_visual),  os.path.join(output_dir, f"{env_name}_pred.gif"),  fps=fps_out)
        log.info(f"  saved 3 gifs to {output_dir}")


if __name__ == "__main__":
    main()
