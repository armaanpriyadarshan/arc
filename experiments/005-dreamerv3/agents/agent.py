"""DreamerV3 agent for ARC-AGI-3.

Ties together the environment wrapper, replay buffer, world model, actor-critic
trainers, and LLM helpers into a single training loop.

Design
------
The agent follows a standard DreamerV3 training flow:

Phase 1  Random exploration — collects ``num_random_episodes`` episodes with
         uniformly random actions to seed the replay buffer before any model
         training begins.

Phase 2  LLM reward shaping — optionally calls Claude once to generate a Python
         reward shaping function tailored to the observed game patterns.

Phase 3  Interleaved training — alternates between collecting real experience
         (every ``collect_interval`` iterations), training the world model on
         replayed sequences, and training the actor-critic in imagination.
         Evaluation runs every ``eval_interval`` iterations. If the agent
         appears stuck (no return improvement for ``llm_diagnosis_patience``
         episodes) it calls Claude for diagnosis.

The agent does NOT inherit from the upstream ``Agent`` base class because that
class has a specific constructor signature (``card_id``, ``arc_env``, etc.)
designed for the reference framework's orchestration layer. Instead, this agent
follows the lighter convention used by experiment 003, where each agent is a
self-contained class with a ``run()`` method.
"""

import logging
import numpy as np
import torch
from typing import Optional

from config import CONFIG
from env_wrapper import ARCEnvWrapper, preprocess_observation
from replay_buffer import ReplayBuffer
from models.world_model import WorldModel
from models.actor import HierarchicalActor
from models.critic import Critic, SlowCritic
from models.rnd import RNDModule
from training.world_model_trainer import WorldModelTrainer
from training.actor_critic_trainer import ActorCriticTrainer
from llm.reward_shaping import generate_reward_shaping_function, default_shaped_reward, ExplorationRewardShaper
from llm.diagnosis import diagnose_stuck_agent
from evaluation.evaluate import evaluate_agent, log_training_stats

logger = logging.getLogger(__name__)


class DreamerAgent:
    """DreamerV3 agent for ARC-AGI-3.

    Args:
        game_id: ARC-AGI-3 game identifier (e.g. ``"ls20"``).
        config: Hyperparameter dict. Defaults to ``CONFIG`` from ``config.py``.
                Any keys present override the defaults.
    """

    def __init__(self, game_id: str, config: Optional[dict] = None) -> None:
        self.game_id = game_id
        self.config = {**CONFIG, **(config or {})}

        # Seeding
        seed = self.config.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("DreamerAgent init: game=%s device=%s", game_id, self.device)

        # ------------------------------------------------------------------ #
        # Environment                                                         #
        # ------------------------------------------------------------------ #
        self.env = ARCEnvWrapper(game_id=game_id)

        # ------------------------------------------------------------------ #
        # Replay buffer                                                        #
        # ------------------------------------------------------------------ #
        self.replay_buffer = ReplayBuffer(
            capacity=self.config["replay_buffer_capacity"]
        )

        # ------------------------------------------------------------------ #
        # Models                                                              #
        # ------------------------------------------------------------------ #
        self.world_model = WorldModel(self.config).to(self.device)

        latent_dim = self.world_model.rssm.latent_dim
        num_actions = self.config["num_base_actions"]
        grid_size   = self.config["grid_size"]

        self.actor = HierarchicalActor(
            latent_dim=latent_dim,
            num_actions=num_actions,
            grid_size=grid_size,
        ).to(self.device)

        self.critic = Critic(latent_dim=latent_dim).to(self.device)
        self.slow_critic = SlowCritic(
            self.critic,
            decay=self.config["critic_slow_ema"],
        )

        # ------------------------------------------------------------------ #
        # Trainers                                                            #
        # ------------------------------------------------------------------ #
        self.wm_trainer = WorldModelTrainer(self.world_model, self.config)
        self.ac_trainer = ActorCriticTrainer(
            self.actor, self.critic, self.slow_critic,
            self.world_model, self.config,
        )

        # ------------------------------------------------------------------ #
        # RND for intrinsic motivation                                        #
        # ------------------------------------------------------------------ #
        self.rnd = RNDModule(in_channels=16, embed_dim=128, lr=1e-3).to(self.device)

        # ------------------------------------------------------------------ #
        # State tracking                                                      #
        # ------------------------------------------------------------------ #
        self.llm_calls: int = 0
        self.episode_returns: list[float] = []
        self._best_trajectory: list[tuple] = []
        self._best_return: float = -float("inf")

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Execute the full DreamerV3 training loop.

        Phases:
            1. Random exploration (fills replay buffer).
            2. Optional LLM reward shaping.
            3. Interleaved training (collect / world model / actor-critic).
        """
        logger.info(
            "Starting DreamerV3: game=%s device=%s max_iterations=%d",
            self.game_id, self.device, self.config["max_iterations"],
        )

        # Attach reward shaping BEFORE exploration so initial data has shaped rewards
        if self.config.get("llm_reward_shaping_enabled", True):
            logger.info("Setting up reward shaping")
            self._setup_reward_shaping()

        # Phase 1: Exploration
        logger.info("Phase 1: exploration (%d episodes)", self.config["num_random_episodes"])
        self._random_exploration()

        # Phase 3: Interleaved training
        logger.info("Phase 3: interleaved training")
        for iteration in range(self.config["max_iterations"]):
            stats = self._training_iteration(iteration)

            if iteration % self.config["log_interval"] == 0:
                log_training_stats(iteration, stats)

            if iteration % self.config["eval_interval"] == 0 and iteration > 0:
                self._evaluate(iteration)

            if self._is_stuck():
                self._llm_diagnosis()

        self.env.close()
        logger.info("DreamerV3 training complete.")

    # ------------------------------------------------------------------ #
    # Phase 1: Random exploration                                          #
    # ------------------------------------------------------------------ #

    def _random_exploration(self) -> None:
        """Collect episodes with structured + random exploration.

        Alternates between:
        - Systematic episodes: go straight until wall, then turn (wall-following)
        - Random episodes: uniform random actions

        This ensures the replay buffer contains both systematic traversals
        (which cover more of the map) and random diversity.
        """
        num_random = self.config["num_random_episodes"]
        max_steps  = self.config["max_episode_steps"]

        for ep in range(num_random):
            obs = self.env.reset()
            # Exclude RESET (index 0) from exploration
            move_actions = [a for a in self.env.valid_action_indices if a != 0]
            if not move_actions:
                move_actions = self.env.valid_action_indices

            observations: list[np.ndarray] = [obs]
            actions:      list[int]   = []
            rewards:      list[float] = []
            dones:        list[bool]  = []

            if ep % 2 == 0:
                # Systematic: wall-following exploration
                current_dir = move_actions[ep % len(move_actions)]
                wall_hits = 0
                for _ in range(max_steps):
                    next_obs, reward, terminated, truncated, _ = self.env.step(current_dir)
                    hit_wall = np.array_equal(obs, next_obs)

                    if hit_wall:
                        wall_hits += 1
                        # Turn: cycle through available directions
                        dir_idx = move_actions.index(current_dir)
                        current_dir = move_actions[(dir_idx + 1) % len(move_actions)]
                        if wall_hits > 8:
                            # Stuck — random jump
                            current_dir = move_actions[np.random.randint(len(move_actions))]
                            wall_hits = 0
                    else:
                        wall_hits = 0

                    observations.append(next_obs)
                    actions.append(current_dir)
                    rewards.append(reward)
                    dones.append(terminated or truncated)
                    obs = next_obs
                    if terminated or truncated:
                        break
            else:
                # Random exploration
                for _ in range(max_steps):
                    action = move_actions[np.random.randint(len(move_actions))]
                    next_obs, reward, terminated, truncated, _ = self.env.step(action)
                    observations.append(next_obs)
                    actions.append(action)
                    rewards.append(reward)
                    dones.append(terminated or truncated)
                    obs = next_obs
                    if terminated or truncated:
                        break

            self.replay_buffer.add_episode(observations, actions, rewards, dones)
            ep_return = sum(rewards)
            logger.info(
                "Explore ep %d/%d (%s): steps=%d return=%.2f",
                ep + 1, num_random,
                "systematic" if ep % 2 == 0 else "random",
                len(actions), ep_return,
            )

    # ------------------------------------------------------------------ #
    # Phase 2: LLM reward shaping                                          #
    # ------------------------------------------------------------------ #

    def _setup_reward_shaping(self) -> None:
        """Generate a reward shaping function from LLM and attach it to the env."""
        try:
            # Extract up to 5 initial frames from the first stored episode
            if not self.replay_buffer.episodes:
                raise ValueError("Replay buffer is empty — cannot extract initial frames")

            first_ep = self.replay_buffer.episodes[0]
            obs_arr = first_ep["observations"]  # (T+1, 64, 64) numpy array
            n_frames = min(5, len(obs_arr))
            initial_frames = [obs_arr[i] for i in range(n_frames)]

            shaped_fn = generate_reward_shaping_function(
                initial_frames,
                game_name=self.game_id,
                model=self.config["llm_model"],
            )
            self.env.set_shaped_reward(shaped_fn)
            self.llm_calls += 1
            logger.info("LLM reward shaping function attached successfully")

        except Exception as exc:
            logger.warning(
                "LLM reward shaping failed (%s) — falling back to exploration shaper", exc
            )
            self.env.set_shaped_reward(ExplorationRewardShaper())

    # ------------------------------------------------------------------ #
    # Phase 3: Training iteration                                          #
    # ------------------------------------------------------------------ #

    def _training_iteration(self, iteration: int) -> dict:
        """One iteration: optionally collect real experience, then train.

        Args:
            iteration: Current iteration index (0-based).

        Returns:
            Dict of training statistics for logging.
        """
        stats: dict = {}

        # Collect a real episode every ``collect_interval`` iterations
        if iteration % self.config["collect_interval"] == 0:
            ep_return = self._collect_episode()
            self.episode_returns.append(ep_return)
            stats["episode_return"] = ep_return

            if ep_return > self._best_return:
                self._best_return = ep_return

        # Train world model on a sampled sequence batch
        batch = self.replay_buffer.sample_sequences(
            batch_size=self.config["batch_size"],
            sequence_length=self.config["sequence_length"],
        )
        wm_stats = self.wm_trainer.train_step(batch)
        stats.update(wm_stats)

        # Train RND predictor on sampled observations
        rnd_obs = self.replay_buffer.sample_states(self.config["batch_size"])
        rnd_onehot = torch.stack(
            [preprocess_observation(rnd_obs[i].numpy()) for i in range(rnd_obs.shape[0])]
        ).to(self.device)
        rnd_loss = self.rnd.train_step(rnd_onehot)
        stats["rnd/loss"] = rnd_loss

        # Train actor-critic in imagination from sampled starting states
        initial_obs = self.replay_buffer.sample_states(self.config["batch_size"])
        ac_stats = self.ac_trainer.train_step(initial_obs)
        stats.update(ac_stats)

        return stats

    def _collect_episode(self) -> float:
        """Collect one episode using the current actor policy.

        Encodes each observation through the world model encoder, maintains
        an RSSM latent state across timesteps, and uses the actor to select
        actions. The episode is stored in the replay buffer.

        Returns:
            Total undiscounted return for the episode.
        """
        max_steps   = self.config["max_episode_steps"]
        num_actions = self.config["num_base_actions"]

        obs = self.env.reset()
        observations: list[np.ndarray] = [obs]
        actions:      list[int]   = []
        rewards:      list[float] = []
        dones:        list[bool]  = []

        # Initialise RSSM state
        h, z = self.world_model.rssm.initial_state(1, self.device)
        prev_action = torch.zeros(1, dtype=torch.long, device=self.device)

        trajectory: list[tuple] = []
        info: dict = {"levels_completed": 0, "state": "NOT_STARTED"}

        self.actor.eval()
        with torch.no_grad():
            for _ in range(max_steps):
                # Encode observation
                obs_tensor = preprocess_observation(obs).unsqueeze(0).to(self.device)
                embed = self.world_model.encoder(obs_tensor)  # (1, embed_dim)

                # Advance RSSM with real observation
                h, z, _, _ = self.world_model.rssm.observe_step(
                    h, z, prev_action, embed
                )
                latent = self.world_model.rssm.get_latent(h, z)  # (1, latent_dim)

                # Sample action
                action_type, x, y, _, _ = self.actor.sample(latent)
                action_int = int(action_type.item())
                x_int = int(x.item())
                y_int = int(y.item())

                next_obs, reward, terminated, truncated, info = self.env.step(
                    action_int, x_int, y_int
                )

                # Add RND intrinsic reward (curiosity bonus)
                next_obs_tensor = preprocess_observation(next_obs).unsqueeze(0).to(self.device)
                rnd_reward = self.rnd.compute_intrinsic_reward(next_obs_tensor).item()
                reward += 0.1 * rnd_reward  # small curiosity bonus

                observations.append(next_obs)
                actions.append(action_int)
                rewards.append(reward)
                done = terminated or truncated
                dones.append(done)
                trajectory.append((obs, action_int, reward))

                obs = next_obs
                prev_action = action_type  # (1,) long tensor

                if done:
                    break

        self.actor.train()

        self.replay_buffer.add_episode(observations, actions, rewards, dones)
        ep_return = sum(rewards)

        # Track best trajectory for LLM diagnosis
        if ep_return > self._best_return:
            self._best_return = ep_return
            self._best_trajectory = trajectory[:]

        levels = info.get("levels_completed", 0)
        logger.info(
            "Collect: steps=%d return=%.2f levels=%d state=%s",
            len(actions), ep_return, levels, info.get("state", "?"),
        )
        return ep_return

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def _evaluate(self, iteration: int) -> None:
        """Run evaluation episodes and log results.

        Args:
            iteration: Current training iteration, used only for logging.
        """
        eval_stats = evaluate_agent(
            env=self.env,
            actor=self.actor,
            world_model=self.world_model,
            preprocess_fn=preprocess_observation,
            num_episodes=3,
            max_steps=self.config["max_episode_steps"],
        )
        logger.info("Eval @ iter %d: %s", iteration, eval_stats)

    # ------------------------------------------------------------------ #
    # Stuck detection and LLM diagnosis                                   #
    # ------------------------------------------------------------------ #

    def _is_stuck(self) -> bool:
        """Detect whether the agent has stopped improving.

        Compares the mean return of the most recent ``patience`` episodes
        against the ``patience`` episodes before that. Returns True when the
        recent mean is no better than the older mean.

        Returns:
            True if the agent appears stuck, False otherwise.
        """
        patience = self.config.get("llm_diagnosis_patience", 100)
        n = len(self.episode_returns)

        if n < patience * 2:
            return False

        recent = np.mean(self.episode_returns[-patience:])
        older  = np.mean(self.episode_returns[-2 * patience:-patience])
        return float(recent) <= float(older)

    def _llm_diagnosis(self) -> None:
        """Call Claude for diagnosis when the agent is stuck.

        Respects the ``llm_max_calls_per_game`` budget. The diagnosis text
        is logged but not acted upon programmatically — it is intended for
        the experimenter to read in ``experiment.log``.
        """
        max_calls = self.config.get("llm_max_calls_per_game", 15)
        if self.llm_calls >= max_calls:
            logger.info(
                "Stuck detected but LLM budget exhausted (%d/%d calls)",
                self.llm_calls, max_calls,
            )
            return

        try:
            diagnosis = diagnose_stuck_agent(
                game_name=self.game_id,
                best_trajectory=self._best_trajectory,
                episode_returns=self.episode_returns,
                model=self.config["llm_model"],
            )
            self.llm_calls += 1
            logger.info("LLM diagnosis (call %d/%d):\n%s", self.llm_calls, max_calls, diagnosis)
        except Exception as exc:
            logger.warning("LLM diagnosis call failed: %s", exc)
