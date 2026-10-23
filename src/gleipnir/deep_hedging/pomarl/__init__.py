"""
JAX-native POMARL (single-agent) deep hedging stack.

Implements Algorithm 1 ("Single-loop AIS-based REINFORCE for POMDP")
from ``notes/pomarl.tex`` against the existing JAX Heston / COS simulator.

See :class:`AISPGTrainer` for the training loop and :class:`PomarlAgent`
for an inference-only adapter compatible with :class:`BaseHedgingAgent`.
"""

from .agent import PomarlAgent
from .ais import AISGRUEncoder, AISRewardModel, AISTransitionModel
from .policy import GaussianPolicy, mean_action, sample_action
from .ppo import ValueCritic, compute_gae, ppo_loss, replay_forward
from .ppo_trainer import PPOTrainer, PPOTrainerConfig
from .rollout import RolloutOutputs, greedy_rollout, stochastic_rollout
from .trainer import AISPGTrainer, AISPGTrainerConfig
from .utils import build_pomdp_obs, discounted_returns, jax_total_payoff

__all__ = [
    "AISGRUEncoder",
    "AISPGTrainer",
    "AISPGTrainerConfig",
    "AISRewardModel",
    "AISTransitionModel",
    "GaussianPolicy",
    "PPOTrainer",
    "PPOTrainerConfig",
    "PomarlAgent",
    "RolloutOutputs",
    "ValueCritic",
    "build_pomdp_obs",
    "compute_gae",
    "discounted_returns",
    "greedy_rollout",
    "jax_total_payoff",
    "mean_action",
    "ppo_loss",
    "replay_forward",
    "sample_action",
    "stochastic_rollout",
]
