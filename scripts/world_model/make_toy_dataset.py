from __future__ import annotations
import argparse
import os
import numpy as np
from world_model.toy_env import Toy2DNavEnv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--steps', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    env = Toy2DNavEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    for ep in range(args.episodes):
        obs = env.reset()
        T = args.steps

        rgb = np.zeros((T, env.H, env.W, 3), dtype=np.uint8)
        depth = np.zeros((T, env.H, env.W), dtype=np.float32)
        imu = np.zeros((T, 6), dtype=np.float32)
        odom = np.zeros((T, 6), dtype=np.float32)   # [x,y,vx,vy,goal_x,goal_y]
        action = np.zeros((T, 2), dtype=np.float32) # [ax,ay]
        collision = np.zeros((T,), dtype=np.float32)

        # Simple goal-seeking policy with noise
        for t in range(T):
            rgb[t] = obs['rgb']
            depth[t] = obs['depth']
            imu[t] = obs['imu']
            odom[t] = obs['odom']
            collision[t] = obs['collision']

            x, y, vx, vy, gx, gy = odom[t]
            to_goal = np.array([gx - x, gy - y], dtype=np.float32)
            d = np.linalg.norm(to_goal) + 1e-6
            dir = to_goal / d
            a = 3.0 * dir + rng.normal(0, 0.8, size=(2,)).astype(np.float32)
            a = np.clip(a, -env.max_accel, env.max_accel)
            action[t] = a

            obs, reward, done, info = env.step(a)
            if done:
                # continue to fill episode with last obs for fixed-length arrays
                pass

        path = os.path.join(args.out_dir, f'ep_{ep:05d}.npz')
        np.savez_compressed(path, rgb=rgb, depth=depth, imu=imu, odom=odom, action=action, collision=collision)

    print(f"Wrote {args.episodes} episodes to {args.out_dir}")

if __name__ == '__main__':
    main()
