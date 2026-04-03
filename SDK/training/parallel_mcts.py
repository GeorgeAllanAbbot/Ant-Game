import math
import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle, ActionCatalog
from SDK.utils.features import FeatureExtractor
from SDK.training.action_encoder import ActionEncoder
from SDK.utils.constants import MAX_ACTIONS

@dataclass
class SearchConfig:
    iterations: int = 64
    max_depth: int = 4
    c_puct: float = 1.25
    root_action_limit: int = 16
    child_action_limit: int = 10
    dirichlet_alpha: float = 0.35
    dirichlet_epsilon: float = 0.25
    seed: int = 0

@dataclass
class MCTSNode:
    state: BackendState
    player: int
    prior: float = 0.0
    action_index: int = 0
    depth: int = 0

    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False

    bundles: List[ActionBundle] = field(default_factory=list)
    priors: Optional[np.ndarray] = None
    children: List['MCTSNode'] = field(default_factory=list)

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

class ParallelMCTS:
    def __init__(self, config: SearchConfig):
        self.config = config
        self.action_catalog = ActionCatalog(max_actions=MAX_ACTIONS)
        self.feature_extractor = FeatureExtractor(max_actions=MAX_ACTIONS)
        self.action_encoder = ActionEncoder(max_actions=MAX_ACTIONS)
        self.rng = random.Random(config.seed)

    def get_action_mask(self, bundles: List[ActionBundle]) -> np.ndarray:
        return self.action_catalog.action_mask(bundles).astype(np.float32)

    def select(self, node: MCTSNode) -> List[MCTSNode]:
        path = [node]
        current = node
        while current.expanded and current.children and current.depth < self.config.max_depth and not current.state.terminal:
            best_child = max(current.children, key=lambda c: self.puct_score(current, c))
            path.append(best_child)
            current = best_child
        return path

    def puct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        explore = self.config.c_puct * child.prior * math.sqrt(parent.visits + 1.0) / (child.visits + 1.0)
        return child.mean_value + explore

    def expand_and_evaluate_request(self, node: MCTSNode) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Returns (board, stats, action_features, action_mask, heuristic_value) for NN batch.
        """
        terminal = self._terminal_value(node.state, node.player)
        if terminal is not None:
            node.expanded = True
            return None, None, None, None, terminal

        node.bundles = self.action_catalog.build(node.state, node.player)
        if not node.bundles or node.depth >= self.config.max_depth:
            node.expanded = True
            return None, None, None, None, 0.0

        mask = self.get_action_mask(node.bundles)
        obs = self.feature_extractor.encode_observation(node.state, node.player, mask)

        board = obs['board']
        stats = obs['stats']
        action_features = self.action_encoder.encode_action(node.bundles, node.player)

        return board, stats, action_features, mask, None

    def apply_nn_evaluation(self, node: MCTSNode, policy: np.ndarray, value: float, is_root: bool = False) -> float:
        node.expanded = True
        node.priors = policy

        if is_root and len(node.bundles) > 1 and self.config.dirichlet_epsilon > 0.0:
            noise = np.random.default_rng(self.rng.randrange(1 << 30)).dirichlet(
                [self.config.dirichlet_alpha] * len(node.bundles)
            ).astype(np.float32)
            prior_slice = policy[:len(node.bundles)]
            prior_slice = (1.0 - self.config.dirichlet_epsilon) * prior_slice + self.config.dirichlet_epsilon * noise

            total = float(np.sum(prior_slice))
            if total > 0:
                prior_slice /= total
            node.priors[:len(node.bundles)] = prior_slice

        limit = self.config.root_action_limit if is_root else self.config.child_action_limit
        branch_indices = self._branch_indices(node.priors, node.bundles, limit)

        for action_index in branch_indices:
            child_state = node.state.clone()
            bundle = node.bundles[action_index]
            enemy_bundles = self.action_catalog.build(child_state, 1 - node.player)
            enemy_bundle = enemy_bundles[0] if enemy_bundles else ActionBundle("hold", 0, ("noop",))

            if node.player == 0:
                child_state.resolve_turn(bundle.operations, enemy_bundle.operations)
            else:
                child_state.resolve_turn(enemy_bundle.operations, bundle.operations)

            node.children.append(
                MCTSNode(
                    state=child_state,
                    player=node.player,
                    prior=float(node.priors[action_index]),
                    action_index=action_index,
                    depth=node.depth + 1
                )
            )

        return value

    def _branch_indices(self, priors: np.ndarray, bundles: List[ActionBundle], limit: int) -> List[int]:
        if not bundles:
            return []
        branch_limit = min(limit, len(bundles))
        order = list(np.argsort(priors[: len(bundles)])[::-1])
        selected = order[:branch_limit]
        if 0 not in selected:
            selected.append(0)
        return sorted(set(int(index) for index in selected))

    def _terminal_value(self, state: BackendState, player: int) -> Optional[float]:
        if not state.terminal:
            return None
        if state.winner is None:
            return 0.0
        return 1.0 if state.winner == player else -1.0

    def backpropagate(self, path: List[MCTSNode], value: float):
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value

    def get_action_probs(self, root: MCTSNode, temperature: float = 1.0) -> Tuple[int, np.ndarray]:
        visit_counts = np.zeros(MAX_ACTIONS, dtype=np.float32)
        for child in root.children:
            visit_counts[child.action_index] = float(child.visits)

        if np.sum(visit_counts) == 0:
            if root.priors is not None:
                visit_counts = root.priors
            else:
                visit_counts[0] = 1.0

        if temperature < 1e-3:
            action = int(np.argmax(visit_counts[:len(root.bundles)]))
            probs = np.zeros_like(visit_counts)
            probs[action] = 1.0
            return action, probs

        # 当温度极小（贪心模式）时，直接使用 Argmax 防止数值溢出
        if temperature < 1e-2:
            best_action_idx = np.argmax(visit_counts)
            probs = np.zeros_like(visit_counts, dtype=np.float32)
            probs[best_action_idx] = 1.0
        else:
            # 正常温度（探索模式）为了绝对安全，加入微小偏置并归一化
            # 避免全 0 访问量时除以 0
            scaled = np.power(visit_counts, 1.0 / temperature)
            sum_scaled = np.sum(scaled)
            if sum_scaled > 0:
                probs = scaled / sum_scaled
            else:
                probs = np.ones_like(visit_counts, dtype=np.float32) / len(visit_counts)

        threshold = self.rng.random()
        cumulative = 0.0
        action = 0
        for i, p in enumerate(probs):
            cumulative += p
            if threshold <= cumulative:
                action = i
                break

        return action, probs

class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, board: np.ndarray, stats: np.ndarray, action_feat: np.ndarray, mask: np.ndarray, policy: np.ndarray, value: float):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (board, stats, action_feat, mask, policy, value)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        boards, stats_batch, action_feats, masks, policies, values = zip(*batch)
        return (
            np.stack(boards),
            np.stack(stats_batch),
            np.stack(action_feats),
            np.stack(masks),
            np.stack(policies),
            np.array(values, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)
