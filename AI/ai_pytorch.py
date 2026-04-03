import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add repo root to path to ensure SDK is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SDK.backend.state import BackendState
from SDK.utils.actions import ActionBundle, ActionCatalog
from SDK.utils.features import FeatureExtractor
from SDK.utils.constants import MAX_ACTIONS
from SDK.training.action_encoder import ActionEncoder
from SDK.training.rl_network import PolicyValueNet

class AI:
    def __init__(self, seed: int = 0, max_actions: int = MAX_ACTIONS):
        self.seed = seed
        self.max_actions = max_actions
        self.action_catalog = ActionCatalog(max_actions=max_actions)
        self.feature_extractor = FeatureExtractor(max_actions=max_actions)
        self.action_encoder = ActionEncoder(max_actions=max_actions)

        # Load the PyTorch model
        self.device = torch.device('cpu')  # Always use CPU for fast inference during competition unless GPU is guaranteed
        self.model = PolicyValueNet()

        # Path to checkpoint. The packaging script should ensure it's placed in 'checkpoints/'
        checkpoint_path = Path(__file__).resolve().parents[1] / 'checkpoints' / 'best_model.pt'

        try:
            if checkpoint_path.exists():
                checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[AI] Successfully loaded PyTorch model from {checkpoint_path}", file=sys.stderr)
            else:
                print(f"[AI Warning] Checkpoint not found at {checkpoint_path}, using untrained weights.", file=sys.stderr)
        except Exception as e:
            print(f"[AI Error] Failed to load model: {e}", file=sys.stderr)

        self.model.eval()

    def choose_bundle(
        self,
        state: BackendState,
        player: int,
        bundles: list[ActionBundle] | None = None,
    ) -> ActionBundle:
        """
        Called every turn by the game engine. Must return exactly one ActionBundle.
        """
        if bundles is None:
            bundles = self.action_catalog.build(state, player)

        if not bundles:
            return ActionBundle("hold", (), 0.0, ("noop",))

        # 1. Generate Masks
        mask = self.action_catalog.action_mask(bundles).astype(np.float32)

        # 2. Extract state representations (board, stats)
        obs = self.feature_extractor.encode_observation(state, player, mask)
        board = obs['board']
        stats = obs['stats']

        # 3. Extract dynamic action features
        action_features = self.action_encoder.encode_action(bundles, player)

        # 4. Neural Network Inference
        with torch.no_grad():
            b = torch.tensor(board, dtype=torch.float32, device=self.device).unsqueeze(0)
            s = torch.tensor(stats, dtype=torch.float32, device=self.device).unsqueeze(0)
            a = torch.tensor(action_features, dtype=torch.float32, device=self.device).unsqueeze(0)
            m = torch.tensor(mask, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Get action probabilities
            policy, value = self.model(b, s, a, m)
            policy = policy[0].numpy()

        # 5. Fallback or selection
        # During competitive play/evaluation, we act greedily (argmax)
        best_index = int(np.argmax(policy[:len(bundles)]))

        return bundles[best_index]
