import numpy as np
from SDK.utils.constants import OperationType, PLAYER_BASES

class ActionEncoder:
    def __init__(self, max_actions: int = 96):
        self.max_actions = max_actions

    def encode_action(self, bundles: list, player: int) -> np.ndarray:
        """
        Creates a dynamic embedding representation for up to `max_actions` ActionBundles.
        Shape: (max_actions, 10)
        Feature map per bundle:
        0: Has Build Operation
        1: Has Upgrade Operation
        2: Has Sell Operation
        3: Has Base Operation
        4: Has Weapon Operation
        5: Normalized target X (if applicable, else 0)
        6: Normalized target Y (if applicable, else 0)
        7: Heuristic Score (Normalized by 50.0)
        8: Is Combo Move (more than 1 operation)
        9: Target Tower Type (Normalized by 43.0, if applicable)
        """
        feats = np.zeros((self.max_actions, 10), dtype=np.float32)

        for i, bundle in enumerate(bundles):
            if i >= self.max_actions:
                break

            has_build = 0.0
            has_upgrade = 0.0
            has_sell = 0.0
            has_base = 0.0
            has_weapon = 0.0
            target_x = 0.0
            target_y = 0.0
            is_combo = 1.0 if len(bundle.operations) > 1 else 0.0
            tower_type = 0.0

            for op in bundle.operations:
                op_type = op.op_type
                if op_type == OperationType.BUILD_TOWER:
                    has_build = 1.0
                    target_x = op.arg0 / 19.0
                    target_y = op.arg1 / 19.0
                elif op_type == OperationType.UPGRADE_TOWER:
                    has_upgrade = 1.0
                    # For upgrade, arg1 is the target tower type
                    tower_type = op.arg1 / 43.0
                elif op_type == OperationType.DOWNGRADE_TOWER:
                    has_sell = 1.0
                elif op_type in (OperationType.UPGRADE_GENERATION_SPEED, OperationType.UPGRADE_GENERATED_ANT):
                    has_base = 1.0
                    # Assign base to its physical location on the grid
                    fb_x, fb_y = PLAYER_BASES[player]
                    target_x = fb_x / 19.0
                    target_y = fb_y / 19.0
                elif op_type in (OperationType.USE_LIGHTNING_STORM, OperationType.USE_EMP_BLASTER, OperationType.USE_DEFLECTOR, OperationType.USE_EMERGENCY_EVASION):
                    has_weapon = 1.0
                    target_x = op.arg0 / 19.0
                    target_y = op.arg1 / 19.0

            feats[i, 0] = has_build
            feats[i, 1] = has_upgrade
            feats[i, 2] = has_sell
            feats[i, 3] = has_base
            feats[i, 4] = has_weapon
            feats[i, 5] = target_x
            feats[i, 6] = target_y
            feats[i, 7] = bundle.score / 50.0 # Will clip to reasonable values later if needed
            feats[i, 8] = is_combo
            feats[i, 9] = tower_type

        return feats
