"""
models/agent.py — Agent REINFORCE dual-network AARLC
=====================================================
Adapté de code/aarlc/ddxplus_code/agent.py.

Architecture fidèle à AARLC :
- Policy network  (sym_acquire_func)  : 4 FC layers → sélection symptôme
- Classifier network (diagnosis_func) : 3 FC layers → diagnostic
- Algorithme : REINFORCE (Monte-Carlo Policy Gradient)
- Loss policy : -log(π(a|s)) × G_t + entropie bonus
- Loss classif : soft_cross_entropy sur diagnostic différentiel
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from scipy.stats import entropy as scipy_entropy


# ─── Réseaux de neurones ─────────────────────────────────────────────────────

class sym_acquire_func(nn.Module):
    """
    Policy network : sélectionne le prochain symptôme à interroger.
    Input: état (D dims) → Output: probabilités sur N_sym symptômes
    Architecture: D → 2048 → 2048 → 2048 → N_sym (Softmax)
    """

    def __init__(self, state_size: int, symptom_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, symptom_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.network(x), dim=-1)


class diagnosis_func(nn.Module):
    """
    Classifier network : prédit la pathologie à partir de l'état courant.
    Input: état (D dims) → Output: logits sur N_patho pathologies
    Architecture: D → 2048 → 2048 → N_patho
    """

    def __init__(self, state_size: int, disease_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, disease_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ─── Loss différentiel ───────────────────────────────────────────────────────

def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropie avec labels souples (distributions cibles).
    Compatible avec les diagnostics différentiels de DDXPlus.

    Loss = -Σ target_i * log(softmax_i)
    """
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(target_probs * log_probs).sum(dim=-1)
    return loss.mean()


# ─── Agent principal ─────────────────────────────────────────────────────────

class Policy_Gradient_pair_model:
    """
    Agent AARLC : REINFORCE avec policy network + classifier network.

    Parameters
    ----------
    state_size : int
        Dimension du vecteur d'état.
    disease_size : int
        Nombre de pathologies.
    symptom_size : int
        Nombre de symptômes (espace d'action).
    LR : float
        Learning rate initial (sera splitté : 2×LR pour classif, LR pour policy).
    Gamma : float
        Facteur de discount.
    device : str
        "cuda" ou "cpu".
    """

    def __init__(
        self,
        state_size: int,
        disease_size: int,
        symptom_size: int,
        LR: float = 1e-4,
        Gamma: float = 0.99,
        device: str = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.gamma = Gamma

        # Réseaux
        self.policy_net = sym_acquire_func(state_size, symptom_size).to(device)
        self.diag_net = diagnosis_func(state_size, disease_size).to(device)

        # Optimiseurs séparés (AARLC : lr différents pour policy vs classif)
        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=LR * 0.2)
        self.optimizer_diag = torch.optim.Adam(self.diag_net.parameters(), lr=LR)

        # Entropie bonus (décroît au fil du temps)
        self.entropy_coeff = 0.01

    def train(self):
        self.policy_net.train()
        self.diag_net.train()

    def eval(self):
        self.policy_net.eval()
        self.diag_net.eval()

    def choose_action_s(self, state: np.ndarray) -> tuple:
        """
        Sélectionne une action (symptôme) via la policy network.

        Returns
        -------
        action : np.ndarray
            Indices des symptômes sélectionnés (batch).
        log_prob : torch.Tensor
            Log-probabilités des actions sélectionnées.
        entropy : torch.Tensor
            Entropie de la distribution de policy.
        """
        s_tensor = torch.FloatTensor(state).to(self.device)
        probs = self.policy_net(s_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        return action.cpu().numpy(), dist.log_prob(action), dist.entropy()

    def choose_diagnosis(self, state: np.ndarray) -> tuple:
        """
        Classifie la pathologie à partir de l'état courant.

        Returns
        -------
        pred : np.ndarray
            Indices des pathologies prédites.
        probs : np.ndarray
            Distributions de probabilités sur les pathologies.
        """
        with torch.no_grad():
            s_tensor = torch.FloatTensor(state).to(self.device)
            logits = self.diag_net(s_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
        pred = np.argmax(probs, axis=-1)
        return pred, probs

    def create_batch(
        self,
        states: list,
        actions: list,
        rewards: list,
    ) -> tuple:
        """
        Calcule les retours G_t cumulés et discompter pour REINFORCE.
        """
        G_list = []
        for ep_rewards in rewards:
            G = 0.0
            ep_G = []
            for r in reversed(ep_rewards):
                G = r + self.gamma * G
                ep_G.insert(0, G)
            G_list.append(ep_G)
        return states, actions, G_list

    def update_param_rl(
        self,
        states: list,
        actions: list,
        returns: list,
        entropy_list: list,
        log_probs_list: list,
        entropy_coeff: float = None,
    ) -> float:
        """
        Met à jour la policy network via REINFORCE.

        Loss = -Σ G_t * log(π(a_t|s_t)) - entropy_coeff * H(π)
        """
        if entropy_coeff is None:
            entropy_coeff = self.entropy_coeff

        all_log_probs = torch.cat(log_probs_list)
        all_returns = torch.FloatTensor(
            [g for ep in returns for g in ep]
        ).to(self.device)
        all_entropy = torch.cat(entropy_list)

        # Normalisation des retours
        if len(all_returns) > 1:
            all_returns = (all_returns - all_returns.mean()) / (all_returns.std() + 1e-8)

        policy_loss = -(all_log_probs * all_returns).mean()
        entropy_loss = -entropy_coeff * all_entropy.mean()
        loss = policy_loss + entropy_loss

        self.optimizer_policy.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer_policy.step()

        return loss.item()

    def update_param_c(
        self,
        states: list,
        target_diffs: list,
    ) -> float:
        """
        Met à jour le classifier network via soft_cross_entropy
        sur les distributions de diagnostic différentiel.
        """
        s_tensor = torch.FloatTensor(np.vstack(states)).to(self.device)
        t_tensor = torch.FloatTensor(np.vstack(target_diffs)).to(self.device)

        logits = self.diag_net(s_tensor)
        loss = soft_cross_entropy(logits, t_tensor)

        self.optimizer_diag.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.diag_net.parameters(), 1.0)
        self.optimizer_diag.step()

        return loss.item()

    def save(self, path: str) -> None:
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "diag_net": self.diag_net.state_dict(),
            "optimizer_policy": self.optimizer_policy.state_dict(),
            "optimizer_diag": self.optimizer_diag.state_dict(),
        }, path)
        print(f"  ✅ Modèle sauvegardé : {path}")

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.diag_net.load_state_dict(checkpoint["diag_net"])
        self.optimizer_policy.load_state_dict(checkpoint["optimizer_policy"])
        self.optimizer_diag.load_state_dict(checkpoint["optimizer_diag"])
        print(f"  ✅ Modèle chargé : {path}")
