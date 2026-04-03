import os
import time
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from queue import Empty

from SDK.backend import create_python_backend_state
from SDK.training.env import AntWarParallelEnv
from SDK.training.rl_network import PolicyValueNet
from SDK.training.parallel_mcts import ParallelMCTS, SearchConfig, MCTSNode, ReplayBuffer
from SDK.utils.constants import MAX_ACTIONS

def mcts_worker(worker_id, request_queue, response_queue):
    config = SearchConfig(seed=worker_id * 1000)
    config.iterations = 64  # 🌟 恢复标准深度，AI 开始展现真正实力
    mcts = ParallelMCTS(config)
    env = AntWarParallelEnv(seed=config.seed, max_actions=MAX_ACTIONS)

    while True:
        obs, infos = env.reset()
        trajectory = []

        for round_idx in range(512):
            if round_idx % 50 == 0:
                print(f"[Worker {worker_id}] Searching Step {round_idx}/512 ...")

            if not env.agents:
                break

            roots = {agent: MCTSNode(state=env.state.clone(), player=i) for i, agent in enumerate(env.possible_agents)}

            for _ in range(config.iterations):
                paths = {}
                batch_boards, batch_stats, batch_action_features, batch_masks, batch_agents = [], [], [], [], []

                for agent in env.possible_agents:
                    root = roots[agent]
                    path = mcts.select(root)
                    leaf = path[-1]
                    paths[agent] = path

                    if not leaf.expanded:
                        board, stats, action_feat, mask, heuristic = mcts.expand_and_evaluate_request(leaf)
                        if heuristic is None:
                            batch_boards.append(board)
                            batch_stats.append(stats)
                            batch_action_features.append(action_feat)
                            batch_masks.append(mask)
                            batch_agents.append(agent)
                        else:
                            mcts.backpropagate(path, heuristic)

                if batch_boards:
                    request_queue.put(('eval', worker_id, np.stack(batch_boards), np.stack(batch_stats), np.stack(batch_action_features), np.stack(batch_masks)))
                    policies, values = response_queue.get()

                    for i, agent in enumerate(batch_agents):
                        path = paths[agent]
                        leaf = path[-1]
                        is_root = (leaf == roots[agent])
                        val = mcts.apply_nn_evaluation(leaf, policies[i], values[i][0], is_root=is_root)
                        mcts.backpropagate(path, val)

            actions = {}
            for i, agent in enumerate(env.possible_agents):
                root = roots[agent]
                if root.bundles:
                    temperature = 1.0 if round_idx < 96 else 1e-3
                    action, probs = mcts.get_action_probs(root, temperature=temperature)
                    actions[agent] = action

                    mask = mcts.get_action_mask(root.bundles)
                    encoded = mcts.feature_extractor.encode_observation(env.state, root.player, mask)
                    action_feat = mcts.action_encoder.encode_action(root.bundles, root.player)
                    trajectory.append((encoded['board'], encoded['stats'], action_feat, mask, probs, root.player))
                else:
                    actions[agent] = 0

            env.step(actions)

        winner = env.state.winner
        final_data = []
        for board, stats, action_feat, mask, probs, player in trajectory:
            val = 0.0
            if winner is not None:
                val = 1.0 if winner == player else -1.0
            final_data.append((board, stats, action_feat, mask, probs, val))

        request_queue.put(('trajectory', final_data))

def evaluator_worker(model_path, best_model_path, result_queue):
    device = torch.device('cpu') 
    current_model = PolicyValueNet().to(device)
    best_model = PolicyValueNet().to(device)
    
    current_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    best_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    current_model.eval()
    best_model.eval()

    config = SearchConfig()
    config.iterations = 64 # 🌟 裁判也恢复最高智商
    mcts_current = ParallelMCTS(config)
    mcts_best = ParallelMCTS(config)

    wins = 0
    draws = 0
    num_games = 20 # 🌟 恢复 20 局严谨评测

    for game in range(num_games):
        env = AntWarParallelEnv(seed=game, max_actions=MAX_ACTIONS)
        env.reset()
        current_side = game % 2

        for round_idx in range(512):
            if not env.agents:
                break

            actions = {}
            for i, agent in enumerate(env.possible_agents):
                bundles = mcts_current.action_catalog.build(env.state, i)
                if not bundles:
                    actions[agent] = 0
                    continue

                mcts = mcts_current if i == current_side else mcts_best
                model = current_model if i == current_side else best_model

                mask = mcts.get_action_mask(bundles)
                obs = mcts.feature_extractor.encode_observation(env.state, i, mask)

                with torch.no_grad():
                    b = torch.tensor(obs['board'], dtype=torch.float32, device=device).unsqueeze(0)
                    s = torch.tensor(obs['stats'], dtype=torch.float32, device=device).unsqueeze(0)
                    a = torch.tensor(mcts.action_encoder.encode_action(bundles, i), dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
                    
                    policy_logits, _ = model(b, s, a, m)
                    policy_logits[~m.bool()] = -1e9
                    policy = F.softmax(policy_logits, dim=1)[0].numpy()

                actions[agent] = int(np.argmax(policy[:len(bundles)]))

            env.step(actions)

        if env.state.winner == current_side:
            wins += 1
        elif env.state.winner is None:
            draws += 1

    win_rate = wins / num_games
    result_queue.put(win_rate)

def safe_save_model(model, path):
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, path)

def train_pipeline():
    mp.set_start_method('spawn', force=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 [PRODUCTION] Initializing Deep Learning Cluster...")

    os.makedirs('checkpoints', exist_ok=True)
    model = PolicyValueNet()

    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        model = nn.DataParallel(model)
    model.to(device)

    pretrained_path = 'checkpoints/sl_pretrained.pth'
    if os.path.exists(pretrained_path):
        try:
            checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            clean_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(clean_dict)
            else:
                model.load_state_dict(clean_dict)
        except Exception as e:
            pass

    safe_save_model(model, 'checkpoints/best_model.pt')
    safe_save_model(model, 'checkpoints/latest_model.pt')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=100000)

    num_workers = 16 # 🌟 满血 16 核心启动
    print(f"⚔️ Spawning {num_workers} MCTS CPU Workers...")
    
    request_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(num_workers)]
    eval_result_queue = mp.Queue()

    workers = []
    for i in range(num_workers):
        p = mp.Process(target=mcts_worker, args=(i, request_queue, response_queues[i]))
        p.start()
        workers.append(p)

    eval_process = None
    step = 0
    batch_boards, batch_stats, batch_action_feats, batch_masks, batch_worker_ids = [], [], [], [], []

    trajectories_collected = 0
    last_trajectories_trained = 0

    try:
        while True:
            # 🌟 恢复 32 的等待队列，完美匹配双卡胃口
            while len(batch_boards) < 32:
                try:
                    msg = request_queue.get(timeout=0.05)
                    if msg[0] == 'trajectory':
                        trajectories_collected += 1
                        for data in msg[1]:
                            buffer.push(*data)
                        print(f"🏁 [Match Finished] Game {trajectories_collected}. Buffer: {len(buffer)}/100000")
                    elif msg[0] == 'eval':
                        _, worker_id, board_np, stats_np, action_feat_np, mask_np = msg
                        for b, s, a, m in zip(board_np, stats_np, action_feat_np, mask_np):
                            batch_boards.append(b)
                            batch_stats.append(s)
                            batch_action_feats.append(a)
                            batch_masks.append(m)
                            batch_worker_ids.append(worker_id)
                except Empty:
                    break

            if batch_boards:
                with torch.no_grad():
                    b = torch.tensor(np.stack(batch_boards), dtype=torch.float32, device=device)
                    st = torch.tensor(np.stack(batch_stats), dtype=torch.float32, device=device)
                    a = torch.tensor(np.stack(batch_action_feats), dtype=torch.float32, device=device)
                    m = torch.tensor(np.stack(batch_masks), dtype=torch.float32, device=device)
                    
                    policies_logits, values = model(b, st, a, m)
                    
                    policies_logits[~m.bool()] = -1e9
                    policies_probs = F.softmax(policies_logits, dim=1)
                    
                    policies = policies_probs.cpu().numpy()
                    values = values.cpu().numpy()

                responses = {w_id: ([], []) for w_id in set(batch_worker_ids)}
                for i, worker_id in enumerate(batch_worker_ids):
                    p_list, v_list = responses[worker_id]
                    p_list.append(policies[i])
                    v_list.append(values[i])

                for worker_id, (p_list, v_list) in responses.items():
                    response_queues[worker_id].put((np.stack(p_list), np.stack(v_list)))

                batch_boards.clear(); batch_stats.clear(); batch_action_feats.clear(); batch_masks.clear(); batch_worker_ids.clear()

            # 🌟 恢复工业级门槛：大于 1024 才开火，每次抽 512
            if len(buffer) >= 1024 and (trajectories_collected - last_trajectories_trained) > 0:
                updates = trajectories_collected - last_trajectories_trained
                last_trajectories_trained = trajectories_collected

                for _ in range(updates):
                    boards, stats, action_feats, masks, policies_target, values_target = buffer.sample(512)

                    b = torch.tensor(boards, dtype=torch.float32, device=device)
                    st = torch.tensor(stats, dtype=torch.float32, device=device)
                    a = torch.tensor(action_feats, dtype=torch.float32, device=device)
                    m = torch.tensor(masks, dtype=torch.float32, device=device)
                    p_target = torch.tensor(policies_target, dtype=torch.float32, device=device)
                    v_target = torch.tensor(values_target, dtype=torch.float32, device=device).unsqueeze(1)

                    optimizer.zero_grad()
                    p_pred_logits, v_pred = model(b, st, a, m)

                    v_loss = F.mse_loss(v_pred, v_target)
                    p_loss = -(p_target * F.log_softmax(p_pred_logits, dim=1)).sum(dim=1).mean()
                    loss = v_loss + p_loss

                    loss.backward()
                    optimizer.step()

                    step += 1

                    if step % 20 == 0:
                        print(f"🔥 [Cluster Train] Step {step} | Value Loss: {v_loss.item():.4f} | Policy Loss: {p_loss.item():.4f}")

                    # 🌟 稳健评测：每 1000 步选拔一次新王
                    if step % 1000 == 0:
                        safe_save_model(model, 'checkpoints/latest_model.pt')
                        if eval_process is None or not eval_process.is_alive():
                            print("⚔️ Dispatching Evaluation Process...")
                            eval_process = mp.Process(
                                target=evaluator_worker,
                                args=('checkpoints/latest_model.pt', 'checkpoints/best_model.pt', eval_result_queue)
                            )
                            eval_process.start()

            # 4. Handle Evaluation Results
            if not eval_result_queue.empty():
                win_rate = eval_result_queue.get()
                print(f"🏆 Evaluation finished! Challenger Win Rate: {win_rate * 100:.2f}%")
                if win_rate > 0.55:
                    print("👑 A NEW KING IS BORN! Updating checkpoints/best_model.pt")
                    os.system("cp checkpoints/latest_model.pt checkpoints/best_model.pt")

    except KeyboardInterrupt:
        print("\n🛑 SHUTTING DOWN CLUSTER GRACEFULLY...")
    finally:
        for w in workers:
            w.terminate()
        if eval_process is not None:
            eval_process.terminate()

if __name__ == '__main__':
    train_pipeline()