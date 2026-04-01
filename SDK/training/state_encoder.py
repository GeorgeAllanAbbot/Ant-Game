import numpy as np
from SDK.backend.state import BackendState
from SDK.utils.constants import MAP_SIZE, MAP_PROPERTY, Terrain, TowerType, AntKind, PLAYER_BASES, MAX_ROUND

class StateEncoder:
    def __init__(self):
        # Cache the obstacle mask since it's static
        self.obstacle_mask = np.zeros((20, 20), dtype=np.float32)

        # Fill obstacles according to MAP_PROPERTY and the 20x20 padding
        for x in range(MAP_SIZE):
            for y in range(MAP_SIZE):
                if MAP_PROPERTY[x][y] == Terrain.VOID:
                    self.obstacle_mask[x, y] = 1.0

        # Pad row 19 and col 19 as obstacles
        self.obstacle_mask[19, :] = 1.0
        self.obstacle_mask[:, 19] = 1.0

    def encode(self, state: BackendState, player: int) -> np.ndarray:
        """
        Converts BackendState to a suitable 4D tensor for neural network input.
        Shape: (12, 20, 20)

        Channels:
        0-1: Friendly/Enemy base locations and current HP percentage.
        2-5: Friendly/Enemy tower distributions (by type ID and HP).
        6-9: Friendly/Enemy ant distributions (density map, normal/combat).
        10: Obstacle mask.
        11: Current coins and round index normalized.
        """
        enemy = 1 - player

        # Initialize tensor with shape (12, 20, 20)
        tensor = np.zeros((12, 20, 20), dtype=np.float32)

        # Channel 0-1: Friendly/Enemy base locations and HP percentage
        fb_x, fb_y = PLAYER_BASES[player]
        eb_x, eb_y = PLAYER_BASES[enemy]

        tensor[0, fb_x, fb_y] = state.bases[player].hp / 50.0
        tensor[1, eb_x, eb_y] = state.bases[enemy].hp / 50.0

        # We need to vectorise this for < 1ms latency

        # Prepare arrays for towers
        if state.towers:
            tower_xs = np.array([t.x for t in state.towers])
            tower_ys = np.array([t.y for t in state.towers])
            tower_players = np.array([t.player for t in state.towers])
            tower_types = np.array([t.type for t in state.towers], dtype=np.float32) / 43.0
            tower_hps = np.array([t.hp / max(t.max_hp, 1) for t in state.towers], dtype=np.float32)

            friendly_towers_mask = (tower_players == player)
            enemy_towers_mask = ~friendly_towers_mask

            f_xs, f_ys = tower_xs[friendly_towers_mask], tower_ys[friendly_towers_mask]
            tensor[2, f_xs, f_ys] = tower_types[friendly_towers_mask]
            tensor[3, f_xs, f_ys] = tower_hps[friendly_towers_mask]

            e_xs, e_ys = tower_xs[enemy_towers_mask], tower_ys[enemy_towers_mask]
            tensor[4, e_xs, e_ys] = tower_types[enemy_towers_mask]
            tensor[5, e_xs, e_ys] = tower_hps[enemy_towers_mask]

        # Prepare arrays for ants
        if state.ants:
            ant_xs = np.array([a.x for a in state.ants])
            ant_ys = np.array([a.y for a in state.ants])
            ant_players = np.array([a.player for a in state.ants])
            ant_kinds = np.array([a.kind for a in state.ants])

            friendly_ants_mask = (ant_players == player)
            enemy_ants_mask = ~friendly_ants_mask

            f_worker_mask = friendly_ants_mask & (ant_kinds == AntKind.WORKER)
            f_combat_mask = friendly_ants_mask & (ant_kinds == AntKind.COMBAT)
            e_worker_mask = enemy_ants_mask & (ant_kinds == AntKind.WORKER)
            e_combat_mask = enemy_ants_mask & (ant_kinds == AntKind.COMBAT)

            np.add.at(tensor[6], (ant_xs[f_worker_mask], ant_ys[f_worker_mask]), 1.0)
            np.add.at(tensor[7], (ant_xs[f_combat_mask], ant_ys[f_combat_mask]), 1.0)
            np.add.at(tensor[8], (ant_xs[e_worker_mask], ant_ys[e_worker_mask]), 1.0)
            np.add.at(tensor[9], (ant_xs[e_combat_mask], ant_ys[e_combat_mask]), 1.0)

        # Channel 10: Obstacle mask (pre-computed)
        tensor[10] = self.obstacle_mask

        # Channel 11: Current coins and round index normalized
        coins_norm = min(state.coins[player] / 500.0, 1.0)
        round_norm = state.round_index / float(MAX_ROUND)
        # Fill the constant values across the whole grid
        tensor[11, :, :] = (coins_norm + round_norm) / 2.0

        return tensor

    def encode_action(self, bundles: list, player: int) -> np.ndarray:
        """
        Creates a dynamic embedding representation for 96 action bundles.
        Shape: (96, 8)
        Feature map per bundle:
        0: Has Build Operation
        1: Has Upgrade Operation
        2: Has Sell Operation
        3: Has Base Operation
        4: Has Weapon Operation
        5: Normalized target X (if applicable, else 0)
        6: Normalized target Y (if applicable, else 0)
        7: Heuristic Score (Normalized roughly)
        """
        feats = np.zeros((96, 8), dtype=np.float32)
        from SDK.utils.constants import OperationType

        for i, bundle in enumerate(bundles):
            if i >= 96:
                break

            has_build = 0.0
            has_upgrade = 0.0
            has_sell = 0.0
            has_base = 0.0
            has_weapon = 0.0
            target_x = 0.0
            target_y = 0.0

            for op in bundle.operations:
                op_type = op.op_type
                if op_type == OperationType.BUILD_TOWER:
                    has_build = 1.0
                    target_x = op.arg0 / 19.0
                    target_y = op.arg1 / 19.0
                elif op_type == OperationType.UPGRADE_TOWER:
                    has_upgrade = 1.0
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
            feats[i, 7] = min(bundle.score / 50.0, 1.0)

        return feats
