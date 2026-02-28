import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models_vision import Encoder, Decoder, TransitionModel, HabitNetwork


def _normalization(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    s = x.sum()
    if float(s.detach().item()) <= eps:
        return torch.ones_like(x) / float(x.numel())
    return x / (s + eps)




def _safe_entropy(probs: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    probs = torch.clamp(probs, min=eps)
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(eps)
    return -(probs * torch.log(probs)).sum(dim=dim)


def kl_diag_normal_with_precision(mu_q, logvar_q, mu_p, logvar_p, omega: torch.Tensor):
    """
    Matches torchutils.kl_div_loss_analytically_from_logvar_and_precision:
      0.5 * (logvar_p - log(omega) - logvar_q)
      + (exp(logvar_q) + (mu_q-mu_p)^2) / (2*exp(logvar_p)/omega)
      - 0.5
    Returns elementwise KL (same shape as mu_q).
    """
    # ensure tensor
    if not torch.is_tensor(omega):
        omega = torch.tensor(float(omega), device=mu_q.device, dtype=mu_q.dtype)
    omega = torch.clamp(omega, min=1e-6)
    return 0.5 * (logvar_p - torch.log(omega) - logvar_q) + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / (
        2.0 * torch.exp(logvar_p) / omega
    ) - 0.5


class MCTSNode:
    """
    Shape-aligned with the original mcts.py Node:
    - stores W (total -G), N (visits), Qpi (habit prior)
    - children as list with length action_dim
    - selection uses Q-normalization + exploration C/N (optionally scaled by Qpi)
    """

    _NODE_ID = 0

    def __init__(self, z_state: torch.Tensor, agent: "DAIMC_Agent", C: float, action_dim: int, using_prior_for_exploration: bool = True):
        self.NODE_ID = MCTSNode._NODE_ID
        MCTSNode._NODE_ID += 1

        self.agent = agent
        self.C = float(C)
        self.action_dim = int(action_dim)
        self.using_prior_for_exploration = bool(using_prior_for_exploration)

        # latent state for this node
        self.z = z_state.detach()

        # statistics (tensor for speed, like original)
        self.W = torch.zeros(self.action_dim, device=self.z.device, dtype=torch.float32)  # total (-G)
        self.N = torch.zeros(self.action_dim, device=self.z.device, dtype=torch.float32)  # visits
        self.Qpi = torch.zeros(self.action_dim, device=self.z.device, dtype=torch.float32)  # habit prior probs

        self.children = [None for _ in range(self.action_dim)]
        self.in_progress = -1  # action index selected on path

        self.expanded = False

    def Q(self):
        return self.W / self.N.clamp_min(1.0)

    def probs_for_selection(self, temperature: float = 1.0):
        # PUCT-style score, sampled via softmax to match paper-style categorical policy use.
        q = self.Q()
        total_n = self.N.sum()
        u = self.C * torch.sqrt(total_n + 1.0) / (1.0 + self.N)
        score = q + (self.Qpi * u if self.using_prior_for_exploration else u)
        if not torch.isfinite(score).all():
            score = torch.nan_to_num(score, nan=0.0, posinf=1e3, neginf=-1e3)
        tau = max(float(temperature), 1e-6)
        return torch.softmax(score / tau, dim=-1)

    def select(self, deterministic: bool = True):
        """
        Returns: path(list of nodes excluding self), actions_path(list of action indices).
        Path[-1] is a leaf candidate (may be unexpanded).
        """
        path = []
        actions_path = []

        # first action
        if deterministic:
            self.in_progress = int(torch.argmax(self.probs_for_selection()).item())
        else:
            self.in_progress = int(torch.multinomial(self.probs_for_selection(), 1).item())
        actions_path.append(self.in_progress)
        child = self.children[self.in_progress]
        path.append(child)

        # descend while node is fully expanded (i.e., all children exist)
        while path[-1] is not None and path[-1].expanded and (None not in path[-1].children):
            node = path[-1]
            if deterministic:
                node.in_progress = int(torch.argmax(node.probs_for_selection()).item())
            else:
                node.in_progress = int(torch.multinomial(node.probs_for_selection(temperature=1.0), 1).item())
            actions_path.append(node.in_progress)
            path.append(node.children[node.in_progress])

        return path, actions_path

    def expand(self, use_means: bool = False, mc_samples: int = 1, init_stats: bool = False):
        """
        Expand *this* node (assumed leaf-ish).
        We compute EFE G for all actions and create all children states (z_next).
        Mirrors the original mcts.py Node.expand behaviour.
        """
        # set habit prior at this node (Qpi)
        with torch.no_grad():
            habit_probs = self.agent.habit_probs_from_z(self.z)
            self.Qpi = habit_probs.detach()

        # compute per-action EFE and next latent
        with torch.no_grad():
            # predict next latent distribution for all actions
            z0 = self.z.unsqueeze(0).repeat(self.action_dim, 1)  # (A, D)
            a_idx = torch.arange(self.action_dim, device=self.z.device, dtype=torch.long)
            a_onehot = F.one_hot(a_idx, num_classes=self.action_dim).float()
            z1_mu, z1_logvar = self.agent.transition(z0, a_onehot, enable_dropout=False)
            if use_means:
                z1 = z1_mu
            else:
                z1 = self.agent.reparameterize(z1_mu, z1_logvar)

            # compute G for each action (vectorized)
            G = self.agent.expected_free_energy_batch(z0, a_idx, z1, mc_samples=mc_samples)  # (A,)
            # Optionally initialize visit/value statistics from immediate one-step EFE estimates.
            # W stores total (-G) so higher is better.
            if init_stats:
                self.N += 1.0
                self.W += (-G).detach()
            # create children nodes
            for a in range(self.action_dim):
                self.children[a] = MCTSNode(z_state=z1[a], agent=self.agent, C=self.C, action_dim=self.action_dim,
                                            using_prior_for_exploration=self.using_prior_for_exploration)

        self.expanded = True

    def backpropagate(self, path_nodes, G_value: torch.Tensor):
        """
        Backpropagate an average G from rollout to the edges along the path.
        Matches original: W[action] -= G ; N[action]+=1
        path_nodes is list of nodes (typically [root] + intermediate_nodes), each must have in_progress set.
        """
        if not torch.is_tensor(G_value):
            G_value = torch.tensor(float(G_value), device=self.z.device)

        for node in path_nodes:
            a = node.in_progress
            if a is None or a < 0:
                raise ValueError("Backpropagate called with invalid in_progress action.")
            node.W[a] -= G_value.detach().float()
            node.N[a] += 1.0
            node.in_progress = -2

    def action_selection(self, deterministic: bool = True):
        """
        Choose a path from root by following the most visited action, as in original.
        Returns list of action indices.
        """
        path = []
        node = self
        while True:
            if deterministic:
                a = int(torch.argmax(node.N).item())
            else:
                a = int(torch.multinomial(node.N.float(), 1).item())
            path.append(a)
            node = node.children[a]
            if node is None or (not node.expanded) or (None in node.children):
                break
        return path


class DAIMC_Agent:
    """
    Deep Active Inference + MCTS, aligned to the original (dsprites) implementation style:
    - selection: Q-normalization + C/N (optionally with habit prior Qpi)
    - value: W stores (-EFE) so higher Q is better
    - rollout: habit-sampled actions
    - training: off-policy minibatch updates (handled in train_deep_agent.py)
    """

    def __init__(self, obs_dim, action_dim, latent_dim=16, aux_obs_dim=2, device=None):
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.aux_obs_dim = int(aux_obs_dim)
        self.device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.encoder = Encoder(self.obs_dim, self.latent_dim, aux_dim=self.aux_obs_dim).to(self.device)
        self.decoder = Decoder(self.latent_dim, self.obs_dim).to(self.device)
        self.transition = TransitionModel(self.latent_dim, self.action_dim).to(self.device)
        self.habit_network = HabitNetwork(self.latent_dim, self.action_dim).to(self.device)
        # Belief head: always predict (normalized) predator distance from latent state.
        # This is trained with an MSE loss against env-provided ground truth in `info["pred_dist_gt"]`.
        self.pred_dist_head = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        ).to(self.device)
        # Optimizers (paper-aligned: learn the generative model + habitual policy)
        self.optimizer_encoder_decoder = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.pred_dist_head.parameters()), lr=5e-4
        )
        self.optimizer_transition = torch.optim.Adam(self.transition.parameters(), lr=1e-4)
        self.optimizer_habit = torch.optim.Adam(self.habit_network.parameters(), lr=1e-4)

        # ---- Hyperparameters matching spirit of original code ----
        # omega schedule params (same shape as torchloss.compute_omega)
        self.omega = torch.tensor(5.0, device=self.device)
        self.omega_a = 4.0
        self.omega_b = 3.0
        self.omega_c = 0.5
        self.omega_d = 1.0

        # gamma mixing between transition prior and naive prior (as in torchloss.compute_loss_down)
        self.gamma = 0.0
        self.gamma_max = 0.8
        self.gamma_warmup_steps = 100000
        self.gamma_warmup_power = 2.0
        # observation / state weighting (like beta_o, beta_s in ModelDown)
        self.beta_o = 50.0
        self.beta_s = 1.0
        # supervised predator-distance loss weight
        self.beta_pred_dist = 1.0

        # MCTS parameters
        self.C = 2.0
        self.using_prior_for_exploration = True

        # MC parameters for EFE term2 estimation
        self.n_theta_samples = 5   # decoder dropout samples

        # Modality scaling / alignment knobs
        # Use sum entropies (paper-like), then explicit scales for numerical balance.
        self.ambiguity_weight = 0.5
        self.vision_entropy_scale = 1.0 / max(1, self.obs_dim)
        # Reconstruction weight for reward observation head (match its importance in preferences)
        self.aux_recon_weight = 1.0
        self.n_o_samples = 2
        self.preference_params = self._init_preferences()
        self.plan_target_mix_alpha = 0.7
        self.plan_target_mix_alpha_min = 0.3
        self.plan_target_mix_decay = 0.999
        self.plan_target_mix_start_step = 20000
        self.plan_target_mix_refresh_stride = 10
        self.apply_omega_to_naive_kl = False
        self.rollout_uniform_mix = 0.10
        self.habit_shortcut_use_entropy = True
        self.habit_shortcut_entropy_threshold = 0.35
        self._last_efe_terms = {}

        self.risk_weight = 1.0
        self.state_info_gain_weight = 0.05
        self.param_info_gain_weight = 1.0


    # ---------------- Preferences (term0) ----------------
    def _init_preferences(self):
        # Bernoulli preference priors over soft observation-features (task-adapted Eq.8a style).
        # Each p_* is a preferred probability for a feature to be "present" (1).
        L = self.obs_dim // 3
        x = torch.linspace(-1.0, 1.0, L, device=self.device)
        center_sigma = 0.28
        center_w = torch.exp(-0.5 * (x / center_sigma) ** 2)
        center_w = center_w / center_w.sum().clamp_min(1e-8)
        return {
            "center_w": center_w.view(1, L),   # (1,L)
            "food_green_thresh": 0.45,
            "pred_red_thresh": 0.45,
            "wall_dark_thresh": 0.18,
            "color_temp": 8.0,
            # Preferred Bernoulli probabilities for feature-level outcomes.
            # We prefer food visible/near center, predator/wall absent, satiated.
            "p_food_center": 0.90,
            "p_food_vis": 0.75,
            "p_pred_center": 0.02,
            "p_pred_vis": 0.05,
            "p_wall_center": 0.08,
            "p_sated": 0.95,      # sated = 1 - hunger            # Relative weights for each Bernoulli preference channel in risk (negative log preference likelihood).
            "w_food_center": 2.0,
            "w_food_vis": 1.0,
            "w_pred_center": 2.5,
            "w_pred_vis": 1.5,
            "w_obstacle_avoid": 2.0,
            "w_hunger_low": 1.5,            "use_hunger_pref": True,            "eps": 1e-6,
            # ---- B안: predator distance preference (farther is better) ----
            "p_pred_far": 0.95,
            "w_pred_far": 2.0,
        }

    # ---------------- Basic ops ----------------

    @staticmethod
    def reparameterize(mu, logvar):
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar).clamp(min=1e-3, max=1e2)
        eps = torch.randn_like(std)
        return mu + eps * std

    
    def infer_z(self, obs):
        """Infer latent state from vision + auxiliary observable (hunger)."""
        if isinstance(obs, dict):
            obs_vis = obs["vision"].to(self.device).float()
            hunger = obs.get("hunger", None)
        else:
            obs_vis = obs.to(self.device).float()
            hunger = None

        if obs_vis.dim() == 1:
            obs_vis = obs_vis.unsqueeze(0)
        elif obs_vis.dim() > 2:
            obs_vis = obs_vis.reshape(obs_vis.shape[0], -1)

        B = obs_vis.shape[0]
        aux = torch.zeros((B, self.aux_obs_dim), device=self.device, dtype=obs_vis.dtype)
        if self.aux_obs_dim >= 1 and hunger is not None:
            if not torch.is_tensor(hunger):
                hunger = torch.tensor(hunger)
            aux[:, 0:1] = hunger.to(self.device).float().view(B, 1).clamp(0.0, 1.0)

        mu, logvar = self.encoder(obs_vis, aux if self.aux_obs_dim > 0 else None)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def decode_obs_dist(self, z: torch.Tensor):
        out = self.decoder(z)
        # Always predict predator distance (normalized to [0,1]) from latent.
        out["pred_dist_hat"] = self.pred_dist_head(z).view(-1)
        return out

    def habit_probs_from_z(self, z: torch.Tensor) -> torch.Tensor:
        logits = self.habit_network(z)
        return F.softmax(logits, dim=-1)

    def _vision_pref_features(self, vision_probs: torch.Tensor):
        """Extract soft preference features from 1D RGB strip observation probabilities."""
        B = vision_probs.shape[0]
        L = self.obs_dim // 3
        x = vision_probs.view(B, L, 3)
        r = x[..., 0]
        g = x[..., 1]
        b = x[..., 2]

        p = self.preference_params
        center_w = p["center_w"]
        temp = float(p["color_temp"])

        brightness = ((r + g + b) / 3.0).clamp(0.0, 1.0)
        green_dom = g - torch.maximum(r, b)
        food_mask = torch.sigmoid(temp * (green_dom - float(p["food_green_thresh"]) * 0.1)).clamp(0.0, 1.0)
        red_dom = r - torch.maximum(g, b)
        pred_mask = torch.sigmoid(temp * (red_dom - float(p["pred_red_thresh"]) * 0.1)).clamp(0.0, 1.0)
        grayness = 1.0 - (torch.abs(r - g) + torch.abs(g - b) + torch.abs(b - r)) / 3.0
        mid_dark = torch.sigmoid(temp * (brightness - 0.03)) * torch.sigmoid(temp * (0.25 - brightness))
        wall_mask = (grayness * mid_dark).clamp(0.0, 1.0)

        cw = center_w.expand(B, -1)
        prox = (0.5 + 0.5 * brightness)
        food_center = (cw * food_mask * prox).sum(dim=-1).clamp(0.0, 1.0)
        pred_center = (cw * pred_mask * prox).sum(dim=-1).clamp(0.0, 1.0)
        wall_center = (cw * wall_mask * prox).sum(dim=-1).clamp(0.0, 1.0)
        food_vis = (food_mask * prox).mean(dim=-1).clamp(0.0, 1.0)
        pred_vis = (pred_mask * prox).mean(dim=-1).clamp(0.0, 1.0)
        danger = (0.7 * pred_center + 0.3 * wall_center).clamp(0.0, 1.0)
        return {
            "food_center": food_center,
            "food_vis": food_vis,
            "pred_vis": pred_vis,
            "pred_center": pred_center,
            "wall_center": wall_center,
            "danger": danger,
        }


    def _observation_preference_term(self, dec_out: dict) -> torch.Tensor:
        """(8a) Risk as negative log-likelihood under Bernoulli preference priors.

        We use soft, decoder-predicted feature probabilities (food/predator/wall + sated)
        and score them against preferred Bernoulli probabilities p_pref. This is closer to the
        paper's preference-prior formulation than a hand-crafted additive penalty.
        """
        p = self.preference_params
        eps = float(p.get("eps", 1e-6))
        vision_probs = dec_out["vision_mean"].clamp(eps, 1.0 - eps)
        vf = self._vision_pref_features(vision_probs)

        def bernoulli_pref_nll(pred_prob: torch.Tensor, pref_prob: float) -> torch.Tensor:
            pref = torch.as_tensor(pref_prob, device=pred_prob.device, dtype=pred_prob.dtype)
            pref = pref.clamp(eps, 1.0 - eps)
            pred_prob = pred_prob.clamp(eps, 1.0 - eps)
            return -(pred_prob * torch.log(pref) + (1.0 - pred_prob) * torch.log(1.0 - pref))

        # Feature-level Bernoulli preference priors (soft probabilities in [0,1]).
        risk = (
            float(p["w_food_center"]) * bernoulli_pref_nll(vf["food_center"], float(p["p_food_center"])) +
            float(p.get("w_food_vis", 1.0)) * bernoulli_pref_nll(vf["food_vis"], float(p.get("p_food_vis", 0.75))) +
            float(p.get("w_pred_center", 2.5)) * bernoulli_pref_nll(vf["pred_center"], float(p.get("p_pred_center", 0.02))) +
            float(p.get("w_pred_vis", 1.5)) * bernoulli_pref_nll(vf["pred_vis"], float(p.get("p_pred_vis", 0.05))) +
            float(p["w_obstacle_avoid"]) * bernoulli_pref_nll(vf["wall_center"], float(p["p_wall_center"]))
        )

        B = vision_probs.shape[0]
        dtype = vision_probs.dtype
        device = vision_probs.device
        if bool(p.get("use_hunger_pref", True)) and ("hunger_pred" in dec_out):
            hunger_prob = dec_out["hunger_pred"].view(-1).clamp(eps, 1.0 - eps)
            sated_prob = (1.0 - hunger_prob).clamp(eps, 1.0 - eps)
            risk = risk + float(p["w_hunger_low"]) * bernoulli_pref_nll(sated_prob, float(p["p_sated"]))
        else:
            risk = risk + torch.zeros(B, device=device, dtype=dtype)

        # Predator-distance preference (B안): farther is better.
        if "pred_dist_hat" in dec_out:
            pred_far = dec_out["pred_dist_hat"].view(-1).clamp(eps, 1.0 - eps)
            risk = risk + float(p.get("w_pred_far", 2.0)) * bernoulli_pref_nll(pred_far, float(p.get("p_pred_far", 0.95)))

        return risk
    @staticmethod
    def _gaussian_diag_entropy_from_logvar(logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        d = logvar.shape[-1]
        return 0.5 * (d * (1.0 + math.log(2.0 * math.pi)) + logvar.sum(dim=-1))

    @staticmethod
    def _bernoulli_entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(1e-6, 1.0 - 1e-6)
        return -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))

    @staticmethod
    def _gaussian_diag_entropy_from_logvar_per_dim(logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return 0.5 * (1.0 + math.log(2.0 * math.pi)) + 0.5 * logvar

    # ---------------- EFE (G) computation ----------------
    def expected_free_energy_batch(self, z0_batch: torch.Tensor, a_idx_batch: torch.Tensor, z1_batch: torch.Tensor, mc_samples: int = 1) -> torch.Tensor:
        """Approximate paper-style EFE with explicit state/parameter epistemic terms.

        G ≈ risk + ambiguity - state_information_gain - parameter_information_gain
        where the latter two correspond to the paper's Eq.(8b)/(8c) concepts.
        """
        dec = self.decode_obs_dist(z1_batch)
        vision_mean = dec["vision_mean"].clamp(1e-6, 1.0 - 1e-6)
        vision_logvar = dec["vision_logvar"]

        # (8a) risk / preference term over predicted observation/state modalities
        risk = self._observation_preference_term(dec)

        # Ambiguity (expected observation entropy): sum entropies + explicit modality scaling.
        H_vision = self._gaussian_diag_entropy_from_logvar_per_dim(vision_logvar).sum(dim=-1) * self.vision_entropy_scale
        ambiguity = self.ambiguity_weight * H_vision

        a_onehot = F.one_hot(a_idx_batch, num_classes=self.action_dim).float()
        _, p_logvar = self.transition(z0_batch, a_onehot, enable_dropout=False)

        # (8b) state epistemic value: reduction in latent-state uncertainty after observing o'.
        state_info_gain = self._term1_state_uncertainty_reduction(z1_batch, p_logvar, mc_samples=mc_samples)
        state_info_gain = torch.clamp(state_info_gain, min=0.0, max=10.0)

        # (8c) parameter epistemic value using MC-dropout over transition+decoder parameters.
        param_info_gain = self._term2_transition_parameter_uncertainty(z0_batch, a_onehot, mc_samples=mc_samples)

        self._last_efe_terms = {
            "risk": float(risk.mean().detach().item()),
            "ambiguity": float(ambiguity.mean().detach().item()),
            "state_info_gain": float(state_info_gain.mean().detach().item()),
            "param_info_gain": float(param_info_gain.mean().detach().item()),
        }
        return (
            self.risk_weight * risk
            + ambiguity
            - self.state_info_gain_weight * state_info_gain
            - self.param_info_gain_weight * param_info_gain
        )

    
    def _term1_state_uncertainty_reduction(
        self,
        z1: torch.Tensor,
        p_logvar: torch.Tensor,
        mc_samples: int = 1,
    ) -> torch.Tensor:
        """Approximate Eq.(8b): latent-state information gain via predicted observations.

        Computes H[p(s'|s,a)] - E_{o'~p(o|s')} H[q(s'|o')].
        Uses predicted vision observations to estimate posterior state uncertainty.
        """
        K = max(1, int(mc_samples))
        with torch.no_grad():
            prior_H = self._gaussian_diag_entropy_from_logvar(p_logvar)

            dec_out = self.decoder(z1)
            vis_mean = dec_out["vision_mean"].clamp(0.0, 1.0)
            vis_std = torch.exp(0.5 * torch.clamp(dec_out["vision_logvar"], min=-8.0, max=4.0))
            aux_obs = None
            if self.aux_obs_dim > 0:
                hunger_hat = dec_out.get("hunger_pred", torch.zeros((z1.shape[0],1), device=z1.device, dtype=z1.dtype))
                aux_obs = hunger_hat[:, :self.aux_obs_dim].clamp(0.0, 1.0)

            post_H = []
            for _ in range(K):
                o_vis = (vis_mean + torch.randn_like(vis_mean) * vis_std).clamp(0.0, 1.0)
                _, q_logvar = self.encoder(o_vis, aux_obs if self.aux_obs_dim > 0 else None)
                post_H.append(self._gaussian_diag_entropy_from_logvar(q_logvar))
            exp_post_H = torch.stack(post_H, dim=0).mean(dim=0)
            return (prior_H - exp_post_H).clamp_min(0.0)

    def _term2_transition_parameter_uncertainty(
        self,
        z0: torch.Tensor,
        a_onehot: torch.Tensor,
        mc_samples: int = 1,
    ) -> torch.Tensor:
        """Approximate Eq.(8c): parameter information gain via MC-dropout over θ.

        Uses dropout samples in both transition and decoder networks.
        """
        T = max(1, int(self.n_theta_samples))
        S = max(1, int(mc_samples))

        with torch.no_grad():
            mi_terms = []
            with self.transition.mc_dropout(), self.decoder.mc_dropout():
                for _ in range(T):
                    mu_t, logvar_t = self.transition(z0, a_onehot, enable_dropout=True)
                    z_theta = torch.stack([self.reparameterize(mu_t, logvar_t).detach() for _ in range(S)], dim=0)  # (S,B,D)
                    probs_theta_s = []
                    ent_theta_s = []
                    for s_idx in range(S):
                        dec_out_t = self.decoder(z_theta[s_idx])
                        m_v = dec_out_t["vision_mean"].clamp(1e-6, 1.0 - 1e-6)
                        lv_v = dec_out_t["vision_logvar"]
                        probs_theta_s.append((m_v, lv_v))
                        H_v = self._gaussian_diag_entropy_from_logvar_per_dim(lv_v).sum(dim=-1) * self.vision_entropy_scale
                        ent_theta_s.append(H_v)
                    m_stack = torch.stack([x[0] for x in probs_theta_s], dim=0)
                    lv_stack = torch.stack([x[1] for x in probs_theta_s], dim=0)
                    m_mean = m_stack.mean(dim=0)
                    second_moment = (torch.exp(lv_stack) + m_stack.pow(2)).mean(dim=0)
                    var_mixture = (second_moment - m_mean.pow(2)).clamp_min(1e-8)
                    lv_mixture = torch.log(var_mixture)
                    H_mean_s = self._gaussian_diag_entropy_from_logvar_per_dim(lv_mixture).sum(dim=-1) * self.vision_entropy_scale
                    H_cond_s = torch.stack(ent_theta_s, dim=0).mean(dim=0)
                    mi_terms.append((m_mean, lv_mixture, H_mean_s, H_cond_s))

            m_across = torch.stack([x[0] for x in mi_terms], dim=0).mean(dim=0)
            lvmix_stack = torch.stack([x[1] for x in mi_terms], dim=0)
            second_across = (torch.exp(lvmix_stack) + torch.stack([x[0] for x in mi_terms], dim=0).pow(2)).mean(dim=0)
            var_across = (second_across - m_across.pow(2)).clamp_min(1e-8)
            H_marginal = self._gaussian_diag_entropy_from_logvar_per_dim(torch.log(var_across)).sum(dim=-1) * self.vision_entropy_scale
            H_cond = torch.stack([x[3] for x in mi_terms], dim=0).mean(dim=0)
            return (H_marginal - H_cond).clamp_min(0.0)

    # ---------------- MCTS planning ----------------

    def plan_action_mcts(self, obs: torch.Tensor, repeats: int = 50, simulation_depth: int = 8, simulation_repeats: int = 3, threshold: float = 0.80, deterministic: bool = True):
        """Paper-style MCTS with optional habitual shortcut and planner visit posterior."""
        modules = [self.encoder, self.decoder, self.transition, self.habit_network]
        prev_modes = [m.training for m in modules]
        try:
            for m in modules:
                m.eval()
            with torch.no_grad():
                if isinstance(obs, dict):
                    obs_in = {k: (v if torch.is_tensor(v) else torch.tensor(v, device=self.device)) for k, v in obs.items()}
                    for k in list(obs_in.keys()):
                        obs_in[k] = obs_in[k].to(self.device).float()
                else:
                    obs_in = obs.to(self.device).float()
                _, z_mu, _ = self.infer_z(obs_in)
                z0 = z_mu[0]
                root = MCTSNode(z_state=z0, agent=self, C=self.C, action_dim=self.action_dim, using_prior_for_exploration=self.using_prior_for_exploration)
                root.Qpi = self.habit_probs_from_z(z0).detach()

                qpi_entropy = float(_safe_entropy(root.Qpi).item())
                qpi_max = float(root.Qpi.max().item())
                use_shortcut = False
                if bool(self.habit_shortcut_use_entropy):
                    use_shortcut = qpi_entropy <= float(self.habit_shortcut_entropy_threshold)
                else:
                    use_shortcut = qpi_max >= float(threshold)

                if use_shortcut:
                    habit_probs = root.Qpi
                    action = int(torch.argmax(habit_probs).item()) if deterministic else int(torch.multinomial(habit_probs, 1).item())
                    return habit_probs.detach().cpu(), action, {"used_habit_shortcut": True, "repeats_used": 0, "states_explored": 0, "root_visit_entropy": qpi_entropy, "root_qpi_max": qpi_max}

                # Expand root once to initialize children and immediate EFE estimates.
                root.expand(use_means=True, mc_samples=max(1, self.n_o_samples), init_stats=True)

                states_explored = 0
                for _ in range(repeats):
                    path, _ = root.select(deterministic=deterministic)
                    leaf = path[-1]
                    if leaf is None:
                        parent = path[-2] if len(path) >= 2 else root
                        # Expand the missing child and continue the same iteration with that child,
                        # so visits/values are updated without "losing" this simulation.
                        parent.expand(use_means=False, mc_samples=max(1, self.n_o_samples))
                        leaf = parent.children[parent.in_progress]
                        if leaf is None:
                            continue
                    if not leaf.expanded:
                        leaf.expand(use_means=False, mc_samples=max(1, self.n_o_samples))

                    rollout_vals = []
                    for _ in range(simulation_repeats):
                        G_rollout, leaf_Qpi = self._simulate_rollout(leaf.z, depth=simulation_depth, use_means=False)
                        states_explored += simulation_depth
                        rollout_vals.append(G_rollout)
                        if leaf_Qpi is not None:
                            leaf.Qpi = leaf_Qpi.detach()
                    G_avg = torch.tensor(rollout_vals, device=self.device).mean() if rollout_vals else torch.tensor(0.0, device=self.device)
                    back_nodes = [root] + [n for n in path[:-1] if n is not None]
                    leaf.backpropagate(back_nodes, G_avg)

                plan_probs = _normalization(root.N)
                action = int(torch.argmax(plan_probs).item()) if deterministic else int(torch.multinomial(plan_probs, 1).item())
                return plan_probs.detach().cpu(), action, {"used_habit_shortcut": False, "repeats_used": repeats, "states_explored": states_explored, "root_visit_entropy": float(_safe_entropy(plan_probs).item()), "root_qpi_max": qpi_max}
        finally:
            for m, was_training in zip(modules, prev_modes):
                m.train(was_training)

    def _simulate_rollout(self, z_start: torch.Tensor, depth: int, use_means: bool = False):
        z = z_start.detach()
        Gs = []
        Qpi_first = None
        for t in range(depth):
            with torch.no_grad():
                probs = self.habit_probs_from_z(z)
                if float(self.rollout_uniform_mix) > 0.0:
                    probs = (1.0 - float(self.rollout_uniform_mix)) * probs + float(self.rollout_uniform_mix) * (torch.ones_like(probs) / probs.numel())
                    probs = probs / probs.sum().clamp_min(1e-8)
                if t == 0:
                    Qpi_first = probs
                a = int(torch.multinomial(probs, 1).item())
                a_onehot = F.one_hot(torch.tensor([a], device=self.device), num_classes=self.action_dim).float()
                mu, logvar = self.transition(z.unsqueeze(0), a_onehot, enable_dropout=False)
                z_next = mu[0] if use_means else self.reparameterize(mu[0], logvar[0])
                G = self.expected_free_energy_batch(z.unsqueeze(0), torch.tensor([a], device=self.device), z_next.unsqueeze(0), mc_samples=max(1, self.n_o_samples))[0]
                Gs.append(G.item())
                z = z_next
        G_avg = float(np.mean(Gs)) if Gs else 0.0
        return G_avg, Qpi_first

    # ---------------- Training losses (off-policy) ----------------
    def compute_habit_kl(self, z: torch.Tensor, plan_probs: torch.Tensor) -> torch.Tensor:
        """
        D_KL[ Q_phi(a|z) || P_plan(a|z) ] (matches torchloss.compute_kl_div_pi style).
        z: (B,D), plan_probs: (B,A)
        """
        logits = self.habit_network(z)
        q = F.softmax(logits, dim=-1)
        log_q = torch.log(torch.clamp(q, min=1e-8))
        log_p = torch.log(torch.clamp(plan_probs, min=1e-8))
        kl = torch.sum(q * (log_q - log_p), dim=-1)
        return kl

    def update_omega(self, kl_pi: torch.Tensor):
        """
        omega = a * (1 - 1/(1+exp(-(kl-b)/c))) + d
        (torchloss.compute_omega)
        """
        kl_mean = kl_pi.mean().detach()
        omega = self.omega_a * (1.0 - 1.0 / (1.0 + torch.exp(-(kl_mean - self.omega_b) / self.omega_c))) + self.omega_d
        self.omega = omega.detach()

    @staticmethod
    def gaussian_diag_nll_per_sample(x: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        inv_var = torch.exp(-logvar)
        nll = 0.5 * (math.log(2.0 * math.pi) + logvar + (x - mean).pow(2) * inv_var)
        return nll.mean(dim=-1)

    def sigmoid_focal_loss_with_logits(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "none",
        pos_weight: torch.Tensor | None = None,
    ):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)  # (B,C)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)                                 # (B,C)

        focal = (1 - p_t).pow(gamma)                                                # (B,C)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)                     # (B,C)

        loss = alpha_t * focal * bce                                                 # (B,C)

        if reduction == "none":
            return loss
        if reduction == "sum":
            return loss.sum()
        if reduction == "mean":
            return loss.mean()
        raise ValueError(reduction)

    def set_gamma_from_global_step(self, global_step: int):
        """Smooth gamma ramp for KL mixing (naive prior -> transition prior)."""
        if global_step is None:
            return float(self.gamma)
        t = max(0.0, min(1.0, float(global_step) / max(1.0, float(self.gamma_warmup_steps))))
        t = t ** float(self.gamma_warmup_power)
        self.gamma = float(self.gamma_max) * t
        return float(self.gamma)

    def update_minibatch(self, obs0, action_idx, obs1, plan_probs, rewards, pred_dist_gt=None,
                         terminateds=None, truncateds=None, sample_steps=None, refresh_plan_targets=False):
        # obs can be tensors or dicts with {vision, hunger}
        if isinstance(obs0, dict):
            obs0_vis = obs0["vision"].to(self.device).float()
            obs0_rew = None
            obs0_hunger = obs0.get("hunger", None)
        else:
            obs0_vis = obs0.to(self.device).float()
            obs0_rew = None
            obs0_hunger = None

        if isinstance(obs1, dict):
            obs1_vis = obs1["vision"].to(self.device).float()
            obs1_rew = None
            obs1_hunger = obs1.get("hunger", None)
        else:
            obs1_vis = obs1.to(self.device).float()
            obs1_rew = None
            obs1_hunger = None

        action_idx = action_idx.to(self.device).long()
        plan_probs = plan_probs.to(self.device).float()
        rewards = rewards.to(self.device).float().view(-1)  # currently not used by the active-inference losses (kept for compatibility)
        if terminateds is None:
            terminateds = torch.zeros_like(rewards, dtype=torch.bool, device=self.device)
        else:
            terminateds = terminateds.to(self.device).bool().view(-1)
        if truncateds is None:
            truncateds = torch.zeros_like(rewards, dtype=torch.bool, device=self.device)
        else:
            truncateds = truncateds.to(self.device).bool().view(-1)
        episode_end = terminateds | truncateds
        # Keep terminated transitions for learning environment dynamics; exclude only time-limit truncations.
        mask_transition = (~truncateds).float()
        mask_habit = (~truncateds).float()
        mask_down = (~truncateds).float()
        den_transition = mask_transition.sum().clamp(min=1.0)
        den_habit = mask_habit.sum().clamp(min=1.0)
        den_down = mask_down.sum().clamp(min=1.0)

        if obs0_vis.dim() > 2:
            obs0_vis = obs0_vis.reshape(obs0_vis.shape[0], -1)
        if obs1_vis.dim() > 2:
            obs1_vis = obs1_vis.reshape(obs1_vis.shape[0], -1)
        obs0_vis = obs0_vis.clamp(0.0, 1.0)

        obs1_vis = obs1_vis.clamp(0.0, 1.0)
        hunger_targets = None
        if obs1_hunger is not None:
            if not torch.is_tensor(obs1_hunger):
                obs1_hunger = torch.tensor(obs1_hunger)
            hunger_targets = obs1_hunger.to(self.device).float().view(-1, 1).clamp(0.0, 1.0)

        # infer latents from vision + auxiliary observations (hunger)
        z0, z0_mu, z0_logvar = self.infer_z({"vision": obs0_vis, "hunger": obs0_hunger})
        z1, z1_mu, z1_logvar = self.infer_z({"vision": obs1_vis, "hunger": obs1_hunger})

        self.optimizer_habit.zero_grad()
        plan_probs_tgt = torch.clamp(plan_probs.detach(), min=1e-8)
        plan_probs_tgt = plan_probs_tgt / plan_probs_tgt.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        can_mix_targets = False
        if refresh_plan_targets:
            if sample_steps is not None:
                current_step_proxy = float(sample_steps.to(self.device).float().max().item())
                can_mix_targets = current_step_proxy >= float(self.plan_target_mix_start_step)
            else:
                can_mix_targets = True

        if refresh_plan_targets and can_mix_targets:
            with torch.no_grad():
                habit_now = self.habit_probs_from_z(z0.detach())
                alpha = float(self.plan_target_mix_alpha)
                plan_probs_tgt = alpha * plan_probs_tgt + (1.0 - alpha) * habit_now
                self.plan_target_mix_alpha = max(self.plan_target_mix_alpha_min, self.plan_target_mix_alpha * self.plan_target_mix_decay)
                plan_probs_tgt = plan_probs_tgt / plan_probs_tgt.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        if sample_steps is not None:
            ages = (sample_steps.to(self.device).float().view(-1).max() - sample_steps.to(self.device).float().view(-1)).clamp_min(0.0)
            stale_w = (1.0 / (1.0 + ages / 5000.0)).detach()
            mask_habit = mask_habit * stale_w
            den_habit = mask_habit.sum().clamp(min=1.0)
        kl_habit_to_plan = self.compute_habit_kl(z0.detach(), plan_probs_tgt)
        habit_loss = (kl_habit_to_plan * mask_habit).sum() / den_habit
        habit_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.habit_network.parameters(), 5.0)
        self.optimizer_habit.step()

        kl_pi = self.compute_habit_kl(z0.detach(), plan_probs_tgt)
        self.update_omega(kl_pi.detach())

        a_onehot = F.one_hot(action_idx, num_classes=self.action_dim).float()
        self.optimizer_transition.zero_grad()
        p_mu, p_logvar = self.transition(z0.detach(), a_onehot.detach(), enable_dropout=False)
        kl_mid = kl_diag_normal_with_precision(z1_mu.detach(), z1_logvar.detach(), p_mu, p_logvar, self.omega).sum(dim=-1)
        loss_mid = (kl_mid * mask_transition).sum() / den_transition
        loss_mid.backward()
        torch.nn.utils.clip_grad_norm_(self.transition.parameters(), 5.0)
        self.optimizer_transition.step()

        self.optimizer_encoder_decoder.zero_grad()
        dec = self.decode_obs_dist(z1)
        vision_mean = dec["vision_mean"]
        vision_logvar = dec["vision_logvar"]
        # Continuous-vision reconstruction aligned with decoder uncertainty (Gaussian NLL).
        recon_vision = self.gaussian_diag_nll_per_sample(obs1_vis, vision_mean, vision_logvar)
        recon_aux = torch.zeros_like(recon_vision)
        recon_hunger = torch.zeros_like(recon_aux)
        if ("hunger_pred" in dec) and (hunger_targets is not None):
            recon_hunger = F.mse_loss(dec["hunger_pred"], hunger_targets, reduction="none").mean(dim=-1)
        recon_aux = recon_hunger
        recon_pix = recon_vision + self.aux_recon_weight * recon_aux

        # ---- B안: supervised predator distance prediction (always trained) ----
        pred_mse = torch.zeros_like(recon_pix)
        if pred_dist_gt is not None:
            if not torch.is_tensor(pred_dist_gt):
                pred_dist_gt = torch.tensor(pred_dist_gt)
            pred_dist_gt_t = pred_dist_gt.to(self.device).float().view(-1).clamp(0.0, 1.0)
            pred_hat = dec["pred_dist_hat"].view(-1).clamp(0.0, 1.0)
            pred_mse = (pred_hat - pred_dist_gt_t).pow(2)
            recon_pix = recon_pix + self.beta_pred_dist * pred_mse

        zeros = torch.zeros_like(z1_mu)
        omega_naive = self.omega if bool(self.apply_omega_to_naive_kl) else torch.tensor(1.0, device=self.device, dtype=z1_mu.dtype)
        kl_naive = kl_diag_normal_with_precision(z1_mu, z1_logvar, zeros, zeros, omega_naive).sum(dim=-1)
        p_mu2, p_logvar2 = self.transition(z0.detach(), a_onehot.detach(), enable_dropout=False)
        kl_trans = kl_diag_normal_with_precision(z1_mu, z1_logvar, p_mu2.detach(), p_logvar2.detach(), self.omega).sum(dim=-1)
        kl_mix = self.gamma * kl_trans + (1.0 - self.gamma) * kl_naive

        F_down = self.beta_o * recon_pix + self.beta_s * kl_mix
        loss_down = (F_down * mask_down).sum() / den_down
        loss_down.backward()
        torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.pred_dist_head.parameters()), 5.0)
        self.optimizer_encoder_decoder.step()

        return {
            "habit_loss": float(habit_loss.item()),
            "kl_pi": float(kl_pi.mean().item()),
            "omega": float(self.omega.item()),
            "loss_mid": float(loss_mid.item()),
            "loss_down": float(loss_down.item()),
            "recon_nll": float(recon_pix.mean().item()),
            "recon_sect": float(recon_vision.mean().item()),
            "recon_aux": float(recon_aux.mean().item()),
            "recon_hunger": float(recon_hunger.mean().item()),
            "pred_dist_mse": float(pred_mse.mean().item()),
                        "kl_naive": float(kl_naive.mean().item()),
            "kl_trans": float(kl_trans.mean().item()),
       
            "efe_risk": float(self._last_efe_terms.get("risk", 0.0)),
            "efe_ambiguity": float(self._last_efe_terms.get("ambiguity", 0.0)),
            "efe_state_info_gain": float(self._last_efe_terms.get("state_info_gain", 0.0)),
            "efe_param_info_gain": float(self._last_efe_terms.get("param_info_gain", 0.0)),
            "plan_target_mix_alpha": float(self.plan_target_mix_alpha),
            "plan_target_mix_applied": float(1.0 if (refresh_plan_targets and can_mix_targets) else 0.0),
            "gamma": float(self.gamma),
            "reward_batch_mean": float(rewards.mean().item()),
 }