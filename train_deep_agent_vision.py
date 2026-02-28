
import os
import random
from collections import deque
from typing import Tuple

import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from zebrafish_env_vision import ZebraFish2DEnv
from agent_vision import DAIMC_Agent


# -----------------------------
# Config
# -----------------------------
SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NUM_EPISODES = 300
MAX_STEPS = 2000

# MCTS knobs (speed/quality tradeoff)
MCTS_REPEATS = 15
SIM_DEPTH = 3
SIM_REPEATS = 1
HABIT_THRESHOLD = 0.80
TIME_SKIP = 1  # kept at 1 because reward is not used in active-inference updates
USE_NSTEP_REWARD_BUFFER = False

# Replay / updates
REPLAY_CAPACITY = 50_000
BATCH_SIZE = 64
UPDATES_PER_EPISODE = 100
WARMUP_STEPS = 1000

# Rendering / recording
RENDER_EVERY_EPISODE = 10
RENDER_DIR = "monitor_videos"
RENDER_FPS = 30
DISPLAY_LIVE = False
DISPLAY_STRIDE = 10


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preprocess_obs(obs):
    """Dict obs -> normalized float32 modalities (vision + optional hunger)."""
    vision = np.asarray(obs["vision"], dtype=np.float32)
    if vision.max() > 1.0:
        vision = vision / 255.0
    vision = vision.reshape(-1).astype(np.float32)
    out = {"vision": vision}
    if "hunger" in obs:
        out["hunger"] = np.asarray(obs["hunger"], dtype=np.float32).reshape(-1)
    return out


def vec_to_strip_rgb(obs_or_vec, scale_h: int = 48) -> np.ndarray:
    if isinstance(obs_or_vec, dict):
        obs_or_vec = obs_or_vec["vision"]
    x = np.asarray(obs_or_vec)
    if x.ndim == 1:
        if x.size % 3 != 0:
            g = x.reshape(1, -1, 1)
            if g.max() <= 1.0:
                g = (255.0 * np.clip(g, 0.0, 1.0)).astype(np.uint8)
            else:
                g = np.clip(g, 0, 255).astype(np.uint8)
            strip = np.repeat(g, 3, axis=2)
        else:
            strip = x.reshape(1, x.size // 3, 3)
            if strip.max() <= 1.0:
                strip = (255.0 * np.clip(strip, 0.0, 1.0)).astype(np.uint8)
            else:
                strip = np.clip(strip, 0, 255).astype(np.uint8)
    else:
        strip = x
        if strip.dtype != np.uint8:
            if strip.max() <= 1.0:
                strip = (255.0 * np.clip(strip, 0.0, 1.0)).astype(np.uint8)
            else:
                strip = np.clip(strip, 0, 255).astype(np.uint8)
    if strip.ndim == 3 and strip.shape[0] == 1:
        strip = np.repeat(strip, scale_h, axis=0)
    return strip


def _repeat_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
    """Nearest-neighbor upsample in width using np.repeat (fallback: index map)."""
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape={img.shape}")
    h, w, c = img.shape
    if w == target_w:
        return img
    if target_w % w == 0:
        return img.repeat(target_w // w, axis=1)
    x_idx = (np.linspace(0, w - 1, target_w)).astype('int64')
    return img[:, x_idx, :]


def _bar_img(value01: float, width: int, height: int, color_rgb: tuple[int, int, int], bg_rgb: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """Create a horizontal bar image (HxWx3) with fill proportional to value01."""
    v = float(np.clip(value01, 0.0, 1.0))
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, :] = np.array(bg_rgb, dtype=np.uint8)
    fill = int(round(v * width))
    if fill > 0:
        img[:, :fill, :] = np.array(color_rgb, dtype=np.uint8)
    return img


def decoder_predictions(agent: DAIMC_Agent, obs_raw) -> tuple[np.ndarray, float, float]:
    """Returns (predicted_vision_strip_rgb, predicted_hunger01, predicted_pred_dist01)."""
    agent.encoder.eval()
    agent.decoder.eval()
    obs_proc = preprocess_obs(obs_raw)
    with torch.no_grad():
        obs_t = {"vision": torch.tensor(obs_proc["vision"], dtype=torch.float32, device=agent.device).unsqueeze(0)}
        if "hunger" in obs_proc:
            obs_t["hunger"] = torch.tensor(obs_proc["hunger"], dtype=torch.float32, device=agent.device).unsqueeze(0)
        z, _, _ = agent.infer_z(obs_t)
        dec = agent.decode_obs_dist(z)
        rec = dec["vision_mean"].squeeze(0).detach().cpu().numpy()
        hunger_hat = float(dec.get("hunger_pred", torch.zeros((1, 1), device=z.device)).view(-1)[0].detach().cpu().item())
        pred_dist_hat = float(dec.get("pred_dist_hat", torch.ones((1,), device=z.device)).view(-1)[0].detach().cpu().item())
    return vec_to_strip_rgb(rec), float(np.clip(hunger_hat, 0.0, 1.0)), float(np.clip(pred_dist_hat, 0.0, 1.0))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=int(capacity))

    def push(self, obs0, a: int, obs1, plan_probs: np.ndarray, reward: float, terminated: bool, truncated: bool,
             step_idx: int = 0, pred_dist_gt: float | None = None):
        if pred_dist_gt is None:
            pred_dist_gt = 1.0
        self.buffer.append((
            np.asarray(obs0["vision"], dtype=np.float32),
            np.asarray(obs0.get("hunger", [0.0]), dtype=np.float32),
            int(a),
            np.asarray(obs1["vision"], dtype=np.float32),
            np.asarray(obs1.get("hunger", [0.0]), dtype=np.float32),
            np.asarray(plan_probs, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            int(step_idx),
            float(pred_dist_gt),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(int(batch_size), len(self.buffer)))
        o0v, o0h, a, o1v, o1h, probs, rew, terminated, truncated, step_idx, pred_dist_gt = zip(*batch)
        return (
            {"vision": np.stack(o0v), "hunger": np.stack(o0h)},
            np.array(a, dtype=np.int64),
            {"vision": np.stack(o1v), "hunger": np.stack(o1h)},
            np.stack(probs),
            np.array(rew, dtype=np.float32),
            np.array(terminated, dtype=np.bool_),
            np.array(truncated, dtype=np.bool_),
            np.array(step_idx, dtype=np.int64),
            np.array(pred_dist_gt, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


def decoder_reconstruction_strip(agent: DAIMC_Agent, obs_raw) -> np.ndarray:
    agent.encoder.eval()
    agent.decoder.eval()
    obs_proc = preprocess_obs(obs_raw)
    with torch.no_grad():
        obs_t = {"vision": torch.tensor(obs_proc["vision"], dtype=torch.float32, device=agent.device).unsqueeze(0)}
        if "hunger" in obs_proc:
            obs_t["hunger"] = torch.tensor(obs_proc["hunger"], dtype=torch.float32, device=agent.device).unsqueeze(0)
        z, _, _ = agent.infer_z(obs_t)
        dec = agent.decode_obs_dist(z)
        rec = dec["vision_mean"].squeeze(0).detach().cpu().numpy()
    return vec_to_strip_rgb(rec)


def record_monitor_episode(
    episode_idx: int,
    agent: DAIMC_Agent,
    max_steps: int,
    seed: int,
    out_dir: str,
    fps: int = 30,
    display: bool = False,
    display_stride: int = 10,
):
    os.makedirs(out_dir, exist_ok=True)
    edge = [
        (0.0, 0.0, 800.0, 1.0),
        (0.0, 0.0, 1.0, 600.0),
        (800.0, 0.0, 1.0, 600.0),
        (0.0, 600.0, 800.0, 1.0),
    ]
    env_vis = ZebraFish2DEnv(render_mode="rgb_array", max_steps=max_steps, n_food=20, obstacles=edge)
    obs, _ = env_vis.reset(seed=seed + episode_idx)

    env_path = os.path.join(out_dir, f"ep{episode_idx:04d}_env.mp4")
    writer_cmp = imageio.get_writer(env_path, fps=fps)

    if display:
        plt.ion()
        plt.figure("ENV", figsize=(6, 4))
        plt.figure("COMPARE", figsize=(8, 4))

    total_reward = 0.0
    done = False
    t = 0
    while not done and t < max_steps:
        obs_proc = preprocess_obs(obs)
        obs_t = {"vision": torch.tensor(obs_proc["vision"], dtype=torch.float32, device=agent.device)}
        if "hunger" in obs_proc:
            obs_t["hunger"] = torch.tensor(obs_proc["hunger"], dtype=torch.float32, device=agent.device)
        plan_probs, action, _ = agent.plan_action_mcts(
            obs_t,
            repeats=MCTS_REPEATS,
            simulation_depth=SIM_DEPTH,
            simulation_repeats=SIM_REPEATS,
            threshold=HABIT_THRESHOLD,
            deterministic=True,
        )

        next_obs, reward, terminated, truncated, _ = env_vis.step(action)
        done = bool(terminated or truncated)
        total_reward += float(reward)

        env_frame = env_vis.render()

        # ---- build stacked comparison frame ----
        # (top)
        # real 1d obs
        # real hunger bar (green)
        # real predator distance bar (red)
        # predicted 1d obs
        # predicted hunger bar (green)
        # predicted predator distance bar (red)
        # real env frame
        # (bottom)
        env_h, env_w = int(env_frame.shape[0]), int(env_frame.shape[1])

        # real modalities
        obs_strip = _repeat_to_width(vec_to_strip_rgb(obs), env_w)
        hunger_gt = float(np.asarray(obs.get("hunger", [0.0]), dtype=np.float32).reshape(-1)[0])

        # pred distance gt from env info
        info_now = env_vis._get_info()
        pred_dist_gt = float(info_now.get("pred_dist_gt", 1.0))

        # predicted modalities
        pred_strip, hunger_hat, pred_dist_hat = decoder_predictions(agent, obs)
        pred_strip = _repeat_to_width(pred_strip, env_w)

        bar_h = 18
        hunger_bar_gt = _bar_img(hunger_gt, env_w, bar_h, color_rgb=(0, 255, 0))
        pred_bar_gt = _bar_img(pred_dist_gt, env_w, bar_h, color_rgb=(255, 0, 0))
        hunger_bar_hat = _bar_img(hunger_hat, env_w, bar_h, color_rgb=(0, 255, 0))
        pred_bar_hat = _bar_img(pred_dist_hat, env_w, bar_h, color_rgb=(255, 0, 0))

        cmp_frame = np.concatenate(
            [
                obs_strip,
                hunger_bar_gt,
                pred_bar_gt,
                pred_strip,
                hunger_bar_hat,
                pred_bar_hat,
                env_frame,
            ],
            axis=0,
        )

        writer_cmp.append_data(cmp_frame)

        if display and (t % max(1, display_stride) == 0):
            plt.figure("ENV")
            plt.clf()
            plt.title(f"Episode {episode_idx} (ENV) t={t}")
            plt.imshow(env_frame)
            plt.axis("off")

            plt.figure("COMPARE")
            plt.clf()
            plt.title(f"Episode {episode_idx} (Obs strip vs decoder) t={t}")
            plt.imshow(cmp_frame)
            plt.axis("off")
            plt.pause(0.001)

        obs = next_obs
        t += 1

    writer_cmp.close()
    env_vis.close()
    if display:
        plt.ioff()
        plt.close("ENV")
        plt.close("COMPARE")

    print(f"[monitor] ep={episode_idx} saved: {env_path} (reward={total_reward:.2f}, steps={t})")


# -----------------------------
# Training
# -----------------------------
def make_agent(env: ZebraFish2DEnv) -> DAIMC_Agent:
    vision_dim = int(np.prod(env.observation_space["vision"].shape))
    aux_obs_dim = 1
    action_dim = int(env.action_space.n)
    return DAIMC_Agent(
        obs_dim=vision_dim,
        aux_obs_dim=aux_obs_dim,
        latent_dim=10,
        action_dim=action_dim,
        device=DEVICE,
    )


def run_train_episode(env_train: ZebraFish2DEnv, agent: DAIMC_Agent, rb: ReplayBuffer, global_step: int) -> Tuple[float, int, int]:
    obs, _ = env_train.reset()
    total_reward = 0.0
    steps = 0
    done = False
    skip_queue = deque()  # optional n-step staging; defaults to 1-step because reward is unused in current loss
    mcts_calls = 0
    habit_shortcuts = 0
    root_entropies = []

    while not done and steps < MAX_STEPS:
        obs_proc = preprocess_obs(obs)
        obs_t = {"vision": torch.tensor(obs_proc["vision"], dtype=torch.float32, device=agent.device)}
        if "hunger" in obs_proc:
            obs_t["hunger"] = torch.tensor(obs_proc["hunger"], dtype=torch.float32, device=agent.device)
        plan_probs_t, action, plan_info = agent.plan_action_mcts(
            obs_t,
            repeats=MCTS_REPEATS,
            simulation_depth=SIM_DEPTH,
            simulation_repeats=SIM_REPEATS,
            threshold=HABIT_THRESHOLD,
            deterministic=False,
        )
        mcts_calls += 1
        if bool(plan_info.get("used_habit_shortcut", False)):
            habit_shortcuts += 1
        if "root_visit_entropy" in plan_info:
            root_entropies.append(float(plan_info["root_visit_entropy"]))
        plan_probs = plan_probs_t.numpy()
        next_obs, reward, terminated, truncated, info = env_train.step(action)
        done = bool(terminated or truncated)
        pred_dist_gt = float(info.get("pred_dist_gt", 1.0))

        next_obs_proc = preprocess_obs(next_obs)
        if USE_NSTEP_REWARD_BUFFER and TIME_SKIP > 1:
            for i in range(len(skip_queue)):
                skip_queue[i][3] += float(reward)            # cumulative reward
                skip_queue[i][4] += 1                        # age / n-step length
                skip_queue[i][6] = next_obs_proc             # per-item next observation
                skip_queue[i][7] = bool(terminated)
                skip_queue[i][8] = bool(truncated)
                skip_queue[i][9] = pred_dist_gt

            # item format:
            # [obs0_proc, action, plan_probs, cum_reward, age, step_idx, next_obs_proc, terminated, truncated]
            skip_queue.append([obs_proc, action, plan_probs, float(reward), 1, global_step, next_obs_proc, bool(terminated), bool(truncated), pred_dist_gt])

            while skip_queue and (skip_queue[0][4] >= TIME_SKIP or done):
                o0_proc, a0, p0, cum_r, age, st0, o1_proc, term0, trunc0, pd0 = skip_queue.popleft()
                rb.push(o0_proc, a0, o1_proc, p0, float(cum_r), bool(term0), bool(trunc0), step_idx=st0, pred_dist_gt=pd0)
        else:
            rb.push(obs_proc, action, next_obs_proc, plan_probs, float(reward), bool(terminated), bool(truncated), step_idx=global_step, pred_dist_gt=pred_dist_gt)

        obs = next_obs
        total_reward += float(reward)
        steps += 1
        global_step += 1

    mcts_stats = {"mcts_calls": mcts_calls, "habit_shortcut_rate": (habit_shortcuts / max(1, mcts_calls)), "root_visit_entropy_mean": (float(np.mean(root_entropies)) if root_entropies else 0.0)}
    return total_reward, steps, global_step, mcts_stats


def train():
    set_seed(SEED)

    edge = [
        (0.0, 0.0, 800.0, 1.0),
        (0.0, 0.0, 1.0, 600.0),
        (800.0, 0.0, 1.0, 600.0),
        (0.0, 600.0, 800.0, 1.0),
    ]
    env_train = ZebraFish2DEnv(render_mode=None, max_steps=MAX_STEPS, n_food=20, obstacles=edge)

    agent = make_agent(env_train)
    rb = ReplayBuffer(REPLAY_CAPACITY)
    os.makedirs(RENDER_DIR, exist_ok=True)

    global_step = 0
    for ep in range(1, NUM_EPISODES + 1):
        ep_reward, ep_steps, global_step, ep_mcts_stats = run_train_episode(env_train, agent, rb, global_step)

        if len(rb) >= max(1, BATCH_SIZE) and global_step >= WARMUP_STEPS:
            for upd in range(UPDATES_PER_EPISODE):
                agent.set_gamma_from_global_step(global_step)
                b_obs0, b_a, b_obs1, b_probs, b_rew, b_terminated, b_truncated, b_steps, b_pred_dist_gt = rb.sample(BATCH_SIZE)
                stats = agent.update_minibatch(
                    {"vision": torch.tensor(b_obs0["vision"], dtype=torch.float32), "hunger": torch.tensor(b_obs0["hunger"], dtype=torch.float32)},
                    torch.tensor(b_a, dtype=torch.long),
                    {"vision": torch.tensor(b_obs1["vision"], dtype=torch.float32), "hunger": torch.tensor(b_obs1["hunger"], dtype=torch.float32)},
                    torch.tensor(b_probs, dtype=torch.float32),
                    torch.tensor(b_rew, dtype=torch.float32),
                    torch.tensor(b_pred_dist_gt, dtype=torch.float32),
                    torch.tensor(b_terminated, dtype=torch.bool),
                    torch.tensor(b_truncated, dtype=torch.bool),
                    sample_steps=torch.tensor(b_steps, dtype=torch.long),
                    refresh_plan_targets=(upd % max(1, int(agent.plan_target_mix_refresh_stride)) == 0),
                )
                if upd % 10 == 0:
                    print(
                        f"[upd] ep={ep:04d} "
                        f"habit_loss={stats['habit_loss']:.2f} "
                        f"kl_pi={stats['kl_pi']:.2f} "
                        f"omega={stats['omega']:.2f} "
                        f"loss_mid={stats['loss_mid']:.2f} "
                        f"loss_down={stats['loss_down']:.2f} "
                        f"recon_nll={stats['recon_nll']:.2f} "
                        f"kl_naive={stats['kl_naive']:.2f} "
                        f"kl_trans={stats['kl_trans']:.2f} "
                        f"efe(r/a/s/p)=({stats['efe_risk']:.2f}/{stats['efe_ambiguity']:.2f}/{stats['efe_state_info_gain']:.2f}/{stats['efe_param_info_gain']:.2f}) "
                        f"gamma={stats['gamma']:.3f} mix_alpha={stats['plan_target_mix_alpha']:.3f} mix_on={int(stats['plan_target_mix_applied'])}"
                    )

        print(f"[train] ep={ep:04d} reward={ep_reward:.2f} steps={ep_steps} buffer={len(rb)} mcts_calls={ep_mcts_stats['mcts_calls']} shortcut_rate={ep_mcts_stats['habit_shortcut_rate']:.2f} rootH={ep_mcts_stats['root_visit_entropy_mean']:.2f}")

        if RENDER_EVERY_EPISODE > 0 and (ep % RENDER_EVERY_EPISODE == 0):
            record_monitor_episode(
                episode_idx=ep,
                agent=agent,
                max_steps=MAX_STEPS,
                seed=SEED,
                out_dir=RENDER_DIR,
                fps=RENDER_FPS,
                display=DISPLAY_LIVE,
                display_stride=DISPLAY_STRIDE,
            )

    env_train.close()
    print("Training complete.")


def main():
    train()


if __name__ == "__main__":
    main()
