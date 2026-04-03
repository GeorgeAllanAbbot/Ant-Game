import numpy as np
import os
import random
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ==========================================
# 导入你的 SDK 和环境组件
# ==========================================
from SDK.training.env import AntWarParallelEnv
from AI.ai_example import AI as GreedyAI 
from SDK.training.action_encoder import ActionEncoder 

def run_single_game(game_seed, device_str):
    """
    单局游戏运行逻辑，供多进程调用
    """
    # 每个进程需要自己的独立环境和 AI 实例
    device = torch.device(device_str)
    env = AntWarParallelEnv() 
    action_encoder = ActionEncoder().to(device) # 🌟 核心：编码器上 GPU
    ai_players = {0: GreedyAI(), 1: GreedyAI()}
    
    # 重置环境
    try:
        obs, info = env.reset(seed=game_seed)
    except:
        obs, info = env.reset()
    
    # 局部 buffer
    g_boards, g_stats, g_masks, g_actions, g_players, g_bundle_features = [], [], [], [], [], []
    
    done = False
    while not done:
        actions_dict = {}
        for player_id, player_obs in obs.items():
            player_info = info[player_id]
            player_int = int(player_id.split('_')[-1]) 
            
            # 提取基础特征
            board = player_obs['board'] 
            stats = player_obs['stats']
            mask = player_obs['action_mask']
            bundles = player_info.get('bundles', None)
            
            # --- 🌟 GPU 加速动作特征提取 ---
            if bundles is not None:
                # 编码并转为 CPU numpy 供存储
                with torch.no_grad():
                    features = action_encoder.encode_action(bundles, player=player_int)
                    current_bundle_features = features.cpu().numpy()
            else:
                # 兜底
                current_bundle_features = np.zeros((96, 10), dtype=np.float32)

            # 获取 backend 状态供专家决策
            backend = getattr(env, 'backend', getattr(env.unwrapped, 'backend', None))
            
            # 专家决策 (带手抖)
            if bundles and backend:
                chosen = ai_players[player_int].choose_bundle(backend, player_int, bundles)
                idx = bundles.index(chosen)
                if random.random() < 0.15 and len(bundles) > 1:
                    top_k = min(3, len(bundles))
                    cands = list(range(top_k))
                    if idx in cands: cands.remove(idx)
                    action_index = random.choice(cands) if cands else idx
                else:
                    action_index = idx
            else:
                valid_ids = np.where(mask)[0]
                action_index = valid_ids[0] if len(valid_ids) > 0 else 0
            
            actions_dict[player_id] = action_index
            
            # 存入局部变量
            g_boards.append(board)
            g_stats.append(stats)
            g_masks.append(mask)
            g_actions.append(action_index)
            g_players.append(player_id)
            g_bundle_features.append(current_bundle_features)
            
        obs, rewards, terms, truncs, info = env.step(actions_dict)
        done = any(terms.values()) or any(truncs.values())

    # 结算胜负
    winner_id = next((pid for pid, rew in rewards.items() if rew > 0), None)
    values = [1.0 if p == winner_id else (-1.0 if winner_id else 0.0) for p in g_players]
    
    return {
        'states': np.array(g_boards), 'stats': np.array(g_stats),
        'masks': np.array(g_masks), 'actions': np.array(g_actions),
        'values': np.array(values), 'features': np.array(g_bundle_features)
    }

def collect_expert_data_mp(num_games=5000, save_path="data/expert_data.npz"):
    print(f"🔥 多进程模式启动！使用 GPU 加速特征提取。目标: {num_games} 局")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 使用 8 个核心（留 2 个给系统和 IO，防止卡死）
    pool_size = min(cpu_count(), 8)
    seeds = [random.randint(0, 999999) for _ in range(num_games)]
    
    results = []
    with Pool(processes=pool_size) as pool:
        # 使用 starmap 进行并行任务分发
        task_args = [(s, device_str) for s in seeds]
        for game_data in tqdm(pool.starmap(run_single_game, task_args), total=num_games):
            results.append(game_data)

    # 合并所有数据
    print("\n📦 正在合并数据并写入硬盘...")
    final_data = {
        'states': np.concatenate([r['states'] for r in results]),
        'stats': np.concatenate([r['stats'] for r in results]),
        'masks': np.concatenate([r['masks'] for r in results]),
        'actions': np.concatenate([r['actions'] for r in results]),
        'values': np.concatenate([r['values'] for r in results]),
        'bundle_features': np.concatenate([r['features'] for r in results])
    }
    
    np.savez_compressed(save_path, **final_data)
    print(f"💾 任务圆满完成！总样本数: {len(final_data['actions'])}")

if __name__ == "__main__":
    collect_expert_data_mp(num_games=5000)