from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class CircleObstacle:
    cx: float
    cy: float
    r: float

@dataclass
class ToyState:
    x: float
    y: float
    vx: float
    vy: float

class Toy2DNavEnv:
    """A tiny continuous 2D environment with circular obstacles.

    State: (x,y,vx,vy) in [-1,1] (position), velocities limited.
    Action: (ax, ay) clipped.

    Observations include:
      - rgb: (H,W,3) uint8 rendering agent/goal/obstacles
      - depth: (H,W) float32 pseudo depth = distance-to-nearest-obstacle field (normalized)
      - odom: vector [x,y,vx,vy, goal_x, goal_y]
      - imu:  vector [ax, ay, vx, vy, 0, 0] (dummy-ish but useful shape)
    """
    def __init__(self, H: int = 64, W: int = 64, dt: float = 0.05, n_obs: int = 4, seed: int = 0):
        self.H, self.W = H, W
        self.dt = dt
        self.rng = np.random.default_rng(seed)
        self.n_obs = n_obs
        self.max_speed = 1.5
        self.max_accel = 4.0
        self.goal = np.zeros(2, dtype=np.float32)
        self.state = ToyState(0,0,0,0)
        self.obstacles: list[CircleObstacle] = []

    def reset(self) -> dict:
        self.obstacles = self._sample_obstacles(self.n_obs)
        self.state = ToyState(*self._sample_free_point(), 0.0, 0.0)
        self.goal = np.array(self._sample_free_point(), dtype=np.float32)
        obs = self._get_obs(last_action=np.zeros(2, dtype=np.float32))
        return obs

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -self.max_accel, self.max_accel)

        # Integrate simple point-mass dynamics
        s = self.state
        vx = np.clip(s.vx + action[0] * self.dt, -self.max_speed, self.max_speed)
        vy = np.clip(s.vy + action[1] * self.dt, -self.max_speed, self.max_speed)
        x = np.clip(s.x + vx * self.dt, -1.0, 1.0)
        y = np.clip(s.y + vy * self.dt, -1.0, 1.0)
        self.state = ToyState(float(x), float(y), float(vx), float(vy))

        # Reward: negative distance to goal + collision penalty
        dist = float(np.linalg.norm(np.array([x,y], dtype=np.float32) - self.goal))
        collision = self._in_collision(x, y)
        reward = -dist - (2.0 if collision else 0.0)
        done = (dist < 0.08) or collision
        info = {"dist_to_goal": dist, "collision": float(collision)}
        obs = self._get_obs(last_action=action)
        return obs, reward, done, info

    def _in_collision(self, x: float, y: float) -> bool:
        for ob in self.obstacles:
            if (x - ob.cx)**2 + (y - ob.cy)**2 <= ob.r**2:
                return True
        return False

    def _sample_obstacles(self, n: int) -> list[CircleObstacle]:
        obs = []
        for _ in range(n):
            cx, cy = self.rng.uniform(-0.7, 0.7, size=2)
            r = float(self.rng.uniform(0.08, 0.18))
            obs.append(CircleObstacle(float(cx), float(cy), r))
        return obs

    def _sample_free_point(self) -> tuple[float, float]:
        for _ in range(2000):
            x, y = self.rng.uniform(-0.9, 0.9, size=2)
            ok = True
            for ob in self.obstacles:
                if (x - ob.cx)**2 + (y - ob.cy)**2 <= (ob.r + 0.08)**2:
                    ok = False
                    break
            if ok:
                return float(x), float(y)
        return 0.0, 0.0

    def _get_obs(self, last_action: np.ndarray) -> dict:
        rgb = self.render_rgb()
        depth = self.render_depth()
        s = self.state
        odom = np.array([s.x, s.y, s.vx, s.vy, float(self.goal[0]), float(self.goal[1])], dtype=np.float32)
        imu = np.array([float(last_action[0]), float(last_action[1]), s.vx, s.vy, 0.0, 0.0], dtype=np.float32)
        collision = float(self._in_collision(s.x, s.y))
        return {"rgb": rgb, "depth": depth, "odom": odom, "imu": imu, "collision": collision}

    def render_rgb(self) -> np.ndarray:
        H, W = self.H, self.W
        img = np.ones((H, W, 3), dtype=np.float32)

        # draw obstacles red-ish
        for ob in self.obstacles:
            self._draw_circle(img, ob.cx, ob.cy, ob.r, color=(1.0, 0.2, 0.2))

        # draw goal green
        self._draw_circle(img, float(self.goal[0]), float(self.goal[1]), 0.06, color=(0.2, 1.0, 0.2))

        # draw agent blue
        s = self.state
        self._draw_circle(img, s.x, s.y, 0.05, color=(0.2, 0.2, 1.0))

        return (np.clip(img, 0, 1) * 255.0).astype(np.uint8)

    def render_depth(self) -> np.ndarray:
        # pseudo depth: per-pixel distance to nearest obstacle boundary (normalized)
        H, W = self.H, self.W
        xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)

        d = np.full((H, W), 2.0, dtype=np.float32)
        for ob in self.obstacles:
            dist_center = np.sqrt((X - ob.cx)**2 + (Y - ob.cy)**2)
            dist_boundary = np.maximum(0.0, dist_center - ob.r)
            d = np.minimum(d, dist_boundary)

        d = np.clip(d / 1.5, 0.0, 1.0)  # normalize
        return d.astype(np.float32)

    def _draw_circle(self, img: np.ndarray, cx: float, cy: float, r: float, color=(1,0,0)) -> None:
        H, W = img.shape[:2]
        # map world [-1,1] to pixel
        px = int((cx + 1.0) * 0.5 * (W - 1))
        py = int((cy + 1.0) * 0.5 * (H - 1))
        pr = int(max(1, r * 0.5 * (W - 1)))
        yy, xx = np.ogrid[:H, :W]
        mask = (xx - px)**2 + (yy - py)**2 <= pr**2
        img[mask, 0] = color[0]
        img[mask, 1] = color[1]
        img[mask, 2] = color[2]
