from __future__ import annotations
import argparse
import numpy as np
import torch
from world_model.models import WorldModel
from world_model.toy_env import Toy2DNavEnv
from world_model.mpc import cem_plan, CEMConfig

def make_cost_fn(w_goal=1.0, w_risk=3.0):
    def cost_fn(state_seq: torch.Tensor, risk_seq: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        # state_seq: (N, H+1, state_dim) state_dim>=2 includes x,y
        # goal: (1,2) [goal_x, goal_y]
        pos = state_seq[..., 0:2]              # (N,H+1,2)
        goal_pos = goal[:, None, :].expand(pos.shape[0], pos.shape[1], 2)
        goal_cost = ((pos - goal_pos) ** 2).sum(dim=-1)  # (N,H+1)
        goal_cost = goal_cost[:, -1]  # terminal
        risk_cost = risk_seq[:, 1:].mean(dim=1)  # average risk over predicted steps
        return w_goal * goal_cost + w_risk * risk_cost
    return cost_fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--horizon', type=int, default=15)
    ap.add_argument('--steps', type=int, default=200)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device)
    train_args = ckpt['args']

    # reconstruct model dims from train args
    model = WorldModel(
        odom_dim=6, 
        imu_dim=6, 
        act_dim=2, 
        state_dim=6,
        backbone=train_args.get('backbone', 'cnn'),
        backbone_kwargs={
            "model_name": train_args.get('backbone_name', 'dinov2_vitb14'),
            "device": args.device
        }
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    env = Toy2DNavEnv(seed=0)
    cfg = CEMConfig(horizon=args.horizon, action_dim=2, action_min=-4.0, action_max=4.0)
    cost_fn = make_cost_fn()

    success = 0
    col = 0
    steps_total = 0

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        for t in range(args.steps):
            # Prepare tensors for encoding
            rgb = torch.from_numpy(obs['rgb']).permute(2,0,1).unsqueeze(0).float().to(args.device) / 255.0
            depth = torch.from_numpy(obs['depth']).unsqueeze(0).unsqueeze(0).float().to(args.device)
            odom = torch.from_numpy(obs['odom']).unsqueeze(0).float().to(args.device)
            imu = torch.from_numpy(obs['imu']).unsqueeze(0).float().to(args.device)
            with torch.no_grad():
                out = model.encode(rgb, depth, odom, imu)
                z0 = out.z

            goal = torch.tensor([[float(obs['odom'][4]), float(obs['odom'][5])]], device=args.device)

            a0, seq = cem_plan(model, z0, goal, cost_fn, cfg, device=torch.device(args.device))
            action = a0.detach().cpu().numpy()

            obs, reward, done, info = env.step(action)
            if info['collision'] > 0.5:
                col += 1
                done = True
            if done:
                if info['dist_to_goal'] < 0.08:
                    success += 1
                steps_total += (t+1)
                break
        if not done:
            steps_total += args.steps

    print(f"Episodes={args.episodes} success={success} collisions={col} avg_steps={steps_total/args.episodes:.1f}")

if __name__ == '__main__':
    main()
