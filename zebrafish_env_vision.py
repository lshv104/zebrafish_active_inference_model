from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


@dataclass
class AgentState:
    pos: np.ndarray        
    heading: float         
    speed: float           


class ZebraFish2DEnv(gym.Env):
    """
    A 2D ZebraFish Environment.
    
    Modifications:
    - Obstacles are now Rectangles: (x, y, width, height)
    - Predator can be toggled on/off.
    - Specific positions can be set for Fish, Predator, Food, and Obstacles.
    """

    metadata = {"render_modes": ["human", "rgb_array", None], "render_fps": 30}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        world_size: Tuple[int, int] = (800, 600),
        max_steps: int = 2000,
        vision_range: float = 350.0,
        fov_deg: float = 120.0,

        initial_fish_pos: Optional[Tuple[float, float]] = None,

        predator_enabled: bool = True,
        initial_pred_pos: Optional[Tuple[float, float]] = None,

        n_food: int = 15,
        initial_food_pos: Optional[List[Tuple[float, float]]] = None,

        obstacles: Optional[List[Tuple[float, float, float, float]]] = None,

        hunger_init: float = 0.8,
        hunger_decay: float = 0.0005,
        hunger_gain_per_food: float = 0.2,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.W, self.H = world_size

        self.max_steps = max_steps
        self.vision_range = float(vision_range)
        self.fov = math.radians(fov_deg)

        self.initial_fish_pos = initial_fish_pos
        self.predator_enabled = predator_enabled
        self.initial_pred_pos = initial_pred_pos
        self.default_n_food = n_food
        self.initial_food_pos = initial_food_pos
        self.fixed_obstacles = obstacles

        self.hunger_init = float(np.clip(hunger_init, 0.0, 1.0))
        self.hunger_decay = float(max(0.0, hunger_decay))
        self.hunger_gain_per_food = float(max(0.0, hunger_gain_per_food))
        self.hunger = self.hunger_init
        self.alive_marker = 1.0

        self.dt = 1.0
        self.max_turn = math.radians(25)
        self.accel = 1.2
        self.max_speed = 8.0
        self.drag = 0.05

        self.fish_r = 14.0
        self.pred_r = 22.0
        self.food_r = 6.0

        self.pred_max_turn = math.radians(10)
        self.pred_speed = 3.0

        # --- Discrete action space (3) ---
        # 0: LEFT            -> (turn=-1, thrust=0)
        # 1: RIGHT           -> (turn=+1, thrust=0)
        # 2: FORWARD         -> (turn=0,  thrust=+1)
        self.n_discrete_actions = 3
        self.action_space = spaces.Discrete(self.n_discrete_actions)

        # Observation Space
        self.vision_1d_width = 160
        self.observation_space = spaces.Dict({
            "vision": spaces.Box(low=0, high=255, shape=(1, self.vision_1d_width, 3), dtype=np.uint8),
            "hunger": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        # preference-related observation channel (state-derived desirability)

        # State containers
        self.steps = 0
        self.fish: AgentState = None
        self.pred: Optional[AgentState] = None
        self.foods: List[np.ndarray] = []
        self.rect_obstacles: List[Tuple[float, float, float, float]] = []

        # render
        self._window = None
        self._clock = None

    def _decode_discrete_action(self, idx: int) -> np.ndarray:
        idx = int(np.clip(idx, 0, self.n_discrete_actions - 1))
        if idx == 0:      # LEFT
            turn, thrust = -1.0, 0.0
        elif idx == 1:    # RIGHT
            turn, thrust = 1.0, 0.0
        else:             # FORWARD
            turn, thrust = 0.0, 1.0
        return np.array([turn, thrust], dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.steps = 0
        self.eaten = 0
        self.hunger = self.hunger_init
        self.alive_marker = 1.0

        # reward obs 초기화 (0 reward -> 0.5)

        # --- 1. Setup Obstacles (Rectangular) ---
        obs_data = self.fixed_obstacles
        if options and "obstacles" in options:
            obs_data = options["obstacles"]

        self.rect_obstacles = []
        if obs_data:
            for o in obs_data:
                self.rect_obstacles.append(tuple(map(float, o)))

        # --- 2. Setup Fish ---
        fish_pos_cfg = self.initial_fish_pos
        if options and "fish_pos" in options:
            fish_pos_cfg = options["fish_pos"]

        if fish_pos_cfg is not None:
            p_fish = np.array(fish_pos_cfg, dtype=np.float32)
        else:
            p_fish = self._rand_pos(margin=60, avoid_obstacles=True, clearance=self.fish_r + 5.0)

        self.fish = AgentState(
            pos=p_fish,
            heading=self.np_random.uniform(-math.pi, math.pi),
            speed=0.0,
        )

        # --- 3. Setup Predator ---
        is_pred_enabled = self.predator_enabled
        if options and "predator_enabled" in options:
            is_pred_enabled = bool(options["predator_enabled"])

        if is_pred_enabled:
            pred_pos_cfg = self.initial_pred_pos
            if options and "pred_pos" in options:
                pred_pos_cfg = options["pred_pos"]

            if pred_pos_cfg is not None:
                p_pred = np.array(pred_pos_cfg, dtype=np.float32)
            else:
                p_pred = self._rand_pos(margin=60, avoid_obstacles=True, clearance=self.pred_r + 5.0)

            self.pred = AgentState(
                pos=p_pred,
                heading=self.np_random.uniform(-math.pi, math.pi),
                speed=self.pred_speed,
            )
        else:
            self.pred = None

        # --- 4. Setup Food ---
        food_pos_list = self.initial_food_pos
        n_food = self.default_n_food

        if options and "food_pos" in options:
            food_pos_list = options["food_pos"]
        elif options and "n_food" in options:
            n_food = options["n_food"]

        self.foods = []
        if food_pos_list is not None:
            for pos in food_pos_list:
                self.foods.append(np.array(pos, dtype=np.float32))
        else:
            self.foods = [
                self._rand_pos(margin=40, avoid_obstacles=True, clearance=self.food_r + 2.0)
                for _ in range(n_food)
            ]

        obs = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return obs, info
    
    def step(self, action):
        self.steps += 1

        # --- Hunger decay (기존 하드코딩 base_decay 제거) ---
        base_decay = float(self.hunger_decay)
        current_speed_ratio = (self.fish.speed / self.max_speed)
        activity_cost = 0.004 * (current_speed_ratio ** 2)
        actual_decay = base_decay + activity_cost
        self.hunger = float(max(0.0, self.hunger - actual_decay))

        # --- (2) Discrete action -> continuous (turn, thrust) ---
        if isinstance(action, (int, np.integer)):
            action_arr = self._decode_discrete_action(int(action))
        else:
            action_arr = np.asarray(action, dtype=np.float32)
            if action_arr.shape == ():
                action_arr = self._decode_discrete_action(int(action_arr.item()))
            action_arr = action_arr.reshape(-1)
            if action_arr.shape[0] < 2:
                raise ValueError(f"Continuous action must have at least 2 elements [turn, thrust], got shape={action_arr.shape}")

        turn = float(np.clip(action_arr[0], -1.0, 1.0))
        thrust = float(np.clip(action_arr[1], -1.0, 1.0))

        # --- Fish Movement ---
        self.fish.heading = self._wrap_angle(self.fish.heading + turn * self.max_turn)
        self.fish.speed = float(np.clip(self.fish.speed + thrust * self.accel, 0.0, self.max_speed))
        self.fish.speed = float(np.clip(self.fish.speed * (1.0 - self.drag), 0.0, self.max_speed))
        self.fish.pos = self._move_with_bounce(self.fish.pos, self.fish.heading, self.fish.speed)
        self._resolve_rect_obstacle_collisions(self.fish, self.fish_r)

        # --- Predator Movement ---
        if self.pred is not None:
            self._predator_policy_step()

        # --- Rewards & termination ---
        reward = -0.001

        newly_eaten = self._eat_food_if_any()
        if newly_eaten > 0:
            reward += 1.0 * newly_eaten
            self.hunger = float(min(1.0, self.hunger + self.hunger_gain_per_food * float(newly_eaten)))

        terminated = False
        truncated = False

        if self.pred is not None:
            caught = self._dist(self.fish.pos, self.pred.pos) <= (self.fish_r + self.pred_r)
            if caught:
                reward -= 10.0
                terminated = True

        if (not terminated) and self.hunger <= 0.0:
            reward -= 5.0
            terminated = True

        if len(self.foods) == 0:
            reward += 5.0
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        # Update preference-related observation markers from current state
        self.alive_marker = 0.0 if terminated else 1.0

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode in ("human", "rgb_array"):
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        try:
            import pygame
        except Exception as e:
            raise DependencyNotInstalled("pygame is not installed. Install it with `pip install pygame`.") from e

        if self._window is None:
            pygame.init()
            if self.render_mode == "human":
                self._window = pygame.display.set_mode((self.W, self.H))
                pygame.display.set_caption("ZebraFish2DEnv")
            else:
                self._window = pygame.Surface((self.W, self.H))
            self._clock = pygame.time.Clock()

        canvas = self._window
        canvas.fill((245, 245, 245))

        # Draw Foods
        for f in self.foods:
            pygame.draw.circle(canvas, (0, 180, 0), f.astype(int), int(self.food_r))

        # Draw Rectangular Obstacles
        for (x, y, w, h) in self.rect_obstacles:
            rect = pygame.Rect(int(x), int(y), int(w), int(h))
            pygame.draw.rect(canvas, (80, 80, 80), rect)

        # Draw Fish
        self._draw_inverted_triangle(
            canvas, self.fish.pos, self.fish.heading, size=(34, 16), 
            color=(40*self.hunger, 80*self.hunger, 220*self.hunger)
        )

        # Draw Predator (if active)
        if self.pred is not None:
            self._draw_inverted_triangle(
                canvas, self.pred.pos, self.pred.heading, size=(48, 22), color=(220, 60, 60)
            )

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()

            pygame.display.flip()
            self._clock.tick(self.metadata.get("render_fps", 30))
            return None

        arr = pygame.surfarray.array3d(canvas)  
        arr = np.transpose(arr, (1, 0, 2))      
        return arr

    def close(self):
        if self._window is not None:
            try:
                import pygame
                pygame.quit()
            except Exception:
                pass
        self._window = None
        self._clock = None

    @staticmethod
    def _ray_circle_intersect(ray_o: np.ndarray, ray_d: np.ndarray, circle_c: np.ndarray, r: float, t_max: float):
        oc = ray_o - circle_c
        b = 2.0 * float(np.dot(oc, ray_d))
        c = float(np.dot(oc, oc)) - float(r) * float(r)
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return None
        sqrt_disc = math.sqrt(disc)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        if 0.0 < t1 <= t_max:
            return float(t1)
        if 0.0 < t2 <= t_max:
            return float(t2)
        return None

    @staticmethod
    def _ray_aabb_intersect(ray_o: np.ndarray, ray_d: np.ndarray, rx: float, ry: float, rw: float, rh: float, t_max: float):
        x_min, x_max = float(rx), float(rx + rw)
        y_min, y_max = float(ry), float(ry + rh)

        def slab(o: float, d: float, mn: float, mx: float):
            if abs(d) < 1e-8:
                if o < mn or o > mx:
                    return None
                return (-float("inf"), float("inf"))
            t1 = (mn - o) / d
            t2 = (mx - o) / d
            return (min(t1, t2), max(t1, t2))

        sx = slab(float(ray_o[0]), float(ray_d[0]), x_min, x_max)
        if sx is None:
            return None
        sy = slab(float(ray_o[1]), float(ray_d[1]), y_min, y_max)
        if sy is None:
            return None

        t_enter = max(sx[0], sy[0])
        t_exit = min(sx[1], sy[1])

        if t_exit < 0.0 or t_enter > t_exit:
            return None
        if 0.0 < t_enter <= t_max:
            return float(t_enter)
        if 0.0 < t_exit <= t_max:
            return float(t_exit)
        return None

    def render_vision_1d(self, width: int = 160, max_range: Optional[float] = None):
        if max_range is None:
            max_range = float(self.vision_range)

        ray_o = self.fish.pos.astype(np.float32)
        fov = float(self.fov)

        img = np.zeros((1, width, 3), dtype=np.uint8)

        bg = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        col_food = np.array([0.0, 180.0, 0.0], dtype=np.float32)
        col_obs = np.array([255.0, 255.0, 255.0], dtype=np.float32)
        col_wall = np.array([255.0, 255.0, 255.0], dtype=np.float32)
        col_pred = np.array([220.0, 60.0, 60.0], dtype=np.float32)

        for i in range(int(width)):
            a = (i + 0.5) / float(width)
            rel = (a - 0.5) * fov
            theta = float(self.fish.heading + rel)
            ray_d = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)

            best_t = float(max_range)
            best_color = bg

            # Food circles
            for f in self.foods:
                t = self._ray_circle_intersect(ray_o, ray_d, f.astype(np.float32), float(self.food_r), best_t)
                if t is not None and t < best_t:
                    best_t = t
                    best_color = col_food

            # Predator circle (if enabled)
            if self.pred is not None:
                t = self._ray_circle_intersect(ray_o, ray_d, self.pred.pos.astype(np.float32), float(self.pred_r), best_t)
                if t is not None and t < best_t:
                    best_t = t
                    best_color = col_pred

            # Rect obstacles
            for (rx, ry, rw, rh) in self.rect_obstacles:
                t = self._ray_aabb_intersect(ray_o, ray_d, rx, ry, rw, rh, best_t)
                if t is not None and t < best_t:
                    best_t = t
                    best_color = col_obs

            # World boundary as AABB
            t = self._ray_aabb_intersect(ray_o, ray_d, 0.0, 0.0, float(self.W), float(self.H), best_t)
            if t is not None and t < best_t:
                best_t = t
                best_color = col_wall

            depth01 = float(np.clip(best_t / float(max_range), 0.0, 1.0))
            shade = 1.0 - 0.85 * depth01
            pix = np.clip(best_color * shade, 0.0, 255.0).astype(np.uint8)

            img[0, i] = pix
        return img

    def _get_obs(self) -> Dict[str, np.ndarray]:
        vision = self.render_vision_1d(width=self.vision_1d_width, max_range=self.vision_range)
        return {"vision": vision, "hunger": np.array([self.hunger], dtype=np.float32)}

    
    def _get_info(self) -> Dict[str, Any]:
        # Always provide predator distance ground-truth (normalized) for supervised belief learning.
        # When predator is disabled, treat it as maximally far (1.0).
        if self.pred is not None:
            d = float(self._dist(self.fish.pos, self.pred.pos))
            max_d = float(math.hypot(self.W, self.H))
            pred_dist_gt = float(np.clip(d / max_d, 0.0, 1.0))
        else:
            pred_dist_gt = 1.0

        return {
            "steps": self.steps,
            "foods_left": len(self.foods),
            "eaten": self.eaten,
            "hunger": float(self.hunger),
            "pred_active": (self.pred is not None),
            "pred_dist_gt": pred_dist_gt,
        }

    def _predator_policy_step(self):
        if self.pred is None:
            return

        to_fish = self.fish.pos - self.pred.pos
        desired = self._angle_to(to_fish)
        delta = self._wrap_angle(desired - self.pred.heading)
        delta = float(np.clip(delta, -self.pred_max_turn, self.pred_max_turn))
        
        self.pred.heading = self._wrap_angle(self.pred.heading + delta)
        self.pred.pos = self._move_with_bounce(self.pred.pos, self.pred.heading, self.pred_speed)
        
        # Predator also respects obstacles
        self._resolve_rect_obstacle_collisions(self.pred, self.pred_r)

    def _eat_food_if_any(self) -> int:
        eaten_idx = None
        for i, f in enumerate(self.foods):
            if self._dist(self.fish.pos, f) <= (self.fish_r + self.food_r):
                eaten_idx = i
                break
        if eaten_idx is None:
            return 0

        self.foods.pop(eaten_idx)
        self.eaten += 1
        return 1

    def _rand_pos(self, margin: float = 30.0, avoid_obstacles: bool = False, clearance: float = 0.0) -> np.ndarray:
        margin = float(max(0.0, margin))
        clearance = float(max(0.0, clearance))

        # Attempt to find a valid position
        for _ in range(3000):
            x = float(self.np_random.uniform(margin, self.W - margin))
            y = float(self.np_random.uniform(margin, self.H - margin))
            
            if avoid_obstacles and self.rect_obstacles:
                conflict = False
                for (rx, ry, rw, rh) in self.rect_obstacles:
                    # Expand rect by clearance for check
                    if (x >= rx - clearance and x <= rx + rw + clearance and
                        y >= ry - clearance and y <= ry + rh + clearance):
                        conflict = True
                        break
                if conflict:
                    continue

            return np.array([x, y], dtype=np.float32)

        # Fallback
        return np.array([self.W/2, self.H/2], dtype=np.float32)

    def _move_with_bounce(self, pos: np.ndarray, heading: float, speed: float) -> np.ndarray:
        v = np.array([math.cos(heading), math.sin(heading)], dtype=np.float32) * speed * self.dt
        new = pos + v

        # Screen boundaries
        if new[0] < 0:
            new[0] = 0
        if new[0] > self.W:
            new[0] = self.W
        if new[1] < 0:
            new[1] = 0
        if new[1] > self.H:
            new[1] = self.H
        return new

    def _resolve_rect_obstacle_collisions(self, agent: AgentState, agent_r: float):
        """
        Collision resolution for Rectangular obstacles.
        Uses closest-point method to push the agent out of the rectangle.
        """
        if not self.rect_obstacles:
            return

        ax, ay = agent.pos[0], agent.pos[1]
        
        for (rx, ry, rw, rh) in self.rect_obstacles:
            # 1. Find the closest point on the AABB to the circle center
            cx = max(rx, min(ax, rx + rw))
            cy = max(ry, min(ay, ry + rh))

            # 2. Distance from closest point to center
            dx = ax - cx
            dy = ay - cy
            dist_sq = dx*dx + dy*dy

            # 3. Check collision
            if dist_sq < (agent_r * agent_r):
                dist = math.sqrt(dist_sq)
                
                # If center is exactly inside (rare), push up
                if dist < 1e-6:
                    nx, ny = 0.0, 1.0
                    depth = agent_r
                else:
                    nx, ny = dx / dist, dy / dist
                    depth = agent_r - dist

                # Push out
                agent.pos[0] += nx * depth
                agent.pos[1] += ny * depth
                
                # Bounce/Slide effect
                # Reflect velocity vector against normal
                v_x = math.cos(agent.heading) * agent.speed
                v_y = math.sin(agent.heading) * agent.speed
                
                dot = v_x * nx + v_y * ny
                
                # If moving into the wall
                if dot < 0:
                    # Slide/Dampen
                    agent.speed *= 0.5 
                    
                    # Reflect heading
                    rx_v = v_x - 2 * dot * nx
                    ry_v = v_y - 2 * dot * ny
                    agent.heading = self._wrap_angle(math.atan2(ry_v, rx_v))

    def _draw_inverted_triangle(self, surface, center: np.ndarray, heading: float, size, color):
        import pygame
        length, width = size

        p_apex = np.array([-length / 2, 0], dtype=np.float32)              
        p_base_left  = np.array([+length / 2, -width / 2], dtype=np.float32)  
        p_base_right = np.array([+length / 2, +width / 2], dtype=np.float32)  

        pts = np.stack([p_apex, p_base_left, p_base_right], axis=0)

        c, s = math.cos(heading), math.sin(heading)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts_w = pts @ R.T + center

        pygame.draw.polygon(surface, color, pts_w.astype(int))

    @staticmethod
    def _dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    @staticmethod
    def _angle_to(vec: np.ndarray) -> float:
        return math.atan2(float(vec[1]), float(vec[0]))

    @staticmethod
    def _wrap_angle(a: float) -> float:
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a

if __name__ == "__main__":
    # Example Usage with new features
    # 1. Custom Obstacles (Rectangles)
    obstacles = [
        (200, 200, 100, 50), # x, y, w, h
        (500, 400, 50, 150)
    ]
    
    # 2. Init with features
    env = ZebraFish2DEnv(
        render_mode="human", 
        obstacles=obstacles,
        n_food=5,             # Default if not overridden in reset
        predator_enabled=True # Default
    )
    
    # 3. Reset with specific overrides
    obs, info = env.reset(
        options={
            "fish_pos": (50, 50),
            "pred_pos": (750, 550),
            "n_food": 10,
            # "predator_enabled": False # Uncomment to disable predator for this episode
        }
    )

    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    env.close()