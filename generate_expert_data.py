import numpy as np
import os
import random
from tqdm import tqdm

# ==========================================
# 导入你的 SDK 和环境组件
# ==========================================
from SDK.training.env import AntWarParallelEnv
from AI.ai_example import AI as GreedyAI 
from SDK.training.action_encoder import ActionEncoder 

def collect_expert_data(num_games=3000, save_path="data/expert_data.npz"):
    print(f"🚀 开始收集专家数据，目标局数: {num_games}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 初始化环境和新版动作编码器
    env = AntWarParallelEnv() 
    action_encoder = ActionEncoder()
    
    ai_players = {0: GreedyAI(), 1: GreedyAI()}
    
    boards_buffer = []    
    stats_buffer = []     
    masks_buffer = []     
    actions_buffer = []   
    values_buffer = []    
    bundle_features_buffer = [] 
    
    fallback_feature_dim = getattr(action_encoder, 'feature_dim', 64)
    
    for game_idx in tqdm(range(num_games), desc="Self-Play Games"):
        # 1. 重置环境 (注入随机种子)
        try:
            obs, info = env.reset(seed=random.randint(0, 99999))
        except TypeError:
            obs, info = env.reset()
        
        game_boards = []
        game_stats = []
        game_masks = []
        game_actions = []
        game_players = [] 
        game_bundle_features = [] 
        
        done = False
        while not done:
            actions_dict = {}
            
            for player_id, player_obs in obs.items():
                player_info = info[player_id]
                player_int = int(player_id.split('_')[-1]) 
                
                # --- a. 提取基础特征 ---
                board_tensor = player_obs['board'] 
                stats_tensor = player_obs['stats']
                action_mask = player_obs['action_mask']
                legal_bundles = player_info.get('bundles', None)
                
                # --- b. 提取 Cross-Attention 需要的动作特征 ---
                if legal_bundles is not None:
                    features = action_encoder.encode_action(legal_bundles, player=player_int)
                    if hasattr(features, 'cpu'): 
                        features = features.detach().cpu().numpy()
                    current_bundle_features = features
                    fallback_feature_dim = current_bundle_features.shape[-1]
                else:
                    current_bundle_features = np.zeros((96, fallback_feature_dim), dtype=np.float32)

                # --- c. 获取 backend_state ---
                backend_state = None
                if hasattr(env, 'backend'):
                    backend_state = env.backend
                elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'backend'):
                    backend_state = env.unwrapped.backend
                elif hasattr(env, 'game'):
                    backend_state = env.game
                
                # --- d. 贪心 AI 做决策 (带 15% 多样性手抖) ---
                if legal_bundles is not None and backend_state is not None:
                    chosen_bundle = ai_players[player_int].choose_bundle(
                        state=backend_state, 
                        player=player_int, 
                        bundles=legal_bundles
                    )
                    expert_action_index = legal_bundles.index(chosen_bundle)
                    
                    if random.random() < 0.15 and len(legal_bundles) > 1:
                        top_k = min(3, len(legal_bundles))
                        candidate_indices = list(range(top_k))
                        if expert_action_index in candidate_indices:
                            candidate_indices.remove(expert_action_index)
                            
                        if len(candidate_indices) > 0:
                            action_index = random.choice(candidate_indices)
                        else:
                            action_index = expert_action_index
                    else:
                        action_index = expert_action_index
                else:
                    valid_indices = np.where(action_mask)[0]
                    if len(valid_indices) > 0:
                        action_index = valid_indices[0] 
                    else:
                        action_index = 0
                        
                actions_dict[player_id] = action_index
                
                # --- e. 记录轨迹 ---
                game_boards.append(board_tensor)
                game_stats.append(stats_tensor)
                game_masks.append(action_mask)
                game_actions.append(action_index)
                game_players.append(player_id)
                game_bundle_features.append(current_bundle_features) 
                
            # 环境步进
            obs, rewards, terminations, truncations, info = env.step(actions_dict)
            
            if isinstance(terminations, dict):
                done = any(terminations.values()) or any(truncations.values())
            else:
                done = terminations or truncations
                
        # 2. 结算胜负
        winner_id = None
        for pid, rew in rewards.items():
            if rew > 0:
                winner_id = pid
                break
                
        # 写入全局 Buffer
        for step_idx in range(len(game_boards)):
            boards_buffer.append(game_boards[step_idx])
            stats_buffer.append(game_stats[step_idx])
            masks_buffer.append(game_masks[step_idx])
            actions_buffer.append(game_actions[step_idx])
            bundle_features_buffer.append(game_bundle_features[step_idx]) 
            
            p = game_players[step_idx]
            if winner_id is None:
                step_value = 0.0
            else:
                step_value = 1.0 if p == winner_id else -1.0
            values_buffer.append(step_value)

    # 3. 保存到硬盘
    print(f"\n✅ 数据收集完毕！总步数(数据量): {len(boards_buffer)}。正在写入硬盘...")
    np.savez_compressed(
        save_path,
        states=np.array(boards_buffer, dtype=np.float32),
        stats=np.array(stats_buffer, dtype=np.float32),
        masks=np.array(masks_buffer, dtype=np.bool_),
        actions=np.array(actions_buffer, dtype=np.int64),
        values=np.array(values_buffer, dtype=np.float32),
        bundle_features=np.array(bundle_features_buffer, dtype=np.float32)
    )
    print(f"💾 数据已保存至 {save_path}")

if __name__ == "__main__":
    collect_expert_data(num_games=2)