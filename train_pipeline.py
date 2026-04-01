import os
import time
import math
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from queue import Empty
from collections import deque
from dataclasses import asdict

from SDK.backend import create_python_backend_state
from SDK.training.env import AntWarParallelEnv
from SDK.training.rl_network import PolicyValueNet
from SDK.training.parallel_mcts import ParallelMCTS, SearchConfig, MCTSNode, ReplayBuffer
from SDK.utils.constants import MAX_ACTIONS

def mcts_worker(worker_id, request_queue, response_queue):
    # This worker runs self-play games continuously
    config = SearchConfig(seed=worker_id * 1000)
    mcts = ParallelMCTS(config)
    env = AntWarParallelEnv(seed=config.seed, max_actions=MAX_ACTIONS)

    while True:
        obs, infos = env.reset()
        trajectory = []

        for round_idx in range(512):
            if not env.agents:
                break

            roots = {agent: MCTSNode(state=env.state.clone(), player=i) for i, agent in enumerate(env.possible_agents)}

            # MCTS simulation loop
            for _ in range(config.iterations):
                paths = {}
                batch_states = []
                batch_action_features = []
                batch_masks = []
                batch_agents = []

                for agent in env.possible_agents:
                    root = roots[agent]
                    # Select
                    path = mcts.select(root)
                    leaf = path[-1]
                    paths[agent] = path

                    if not leaf.expanded:
                        # Expand
                        encoded, action_feat, mask, heuristic = mcts.expand_and_evaluate_request(leaf)
                        if heuristic is None:
                            # Needs NN evaluation
                            batch_states.append(encoded)
                            batch_action_features.append(action_feat)
                            batch_masks.append(mask)
                            batch_agents.append(agent)
                        else:
                            # Terminal state
                            mcts.backpropagate(path, heuristic)

                # If we have nodes that need NN eval, send batch request
                if batch_states:
                    state_np = np.stack(batch_states)
                    action_feat_np = np.stack(batch_action_features)
                    mask_np = np.stack(batch_masks)
                    request_queue.put(('eval', worker_id, state_np, action_feat_np, mask_np))

                    # Wait for response
                    policies, values = response_queue.get()

                    for i, agent in enumerate(batch_agents):
                        path = paths[agent]
                        leaf = path[-1]
                        is_root = (leaf == roots[agent])
                        val = mcts.apply_nn_evaluation(leaf, policies[i], values[i][0], is_root=is_root)
                        mcts.backpropagate(path, val)

            # Action selection after MCTS completes
            actions = {}
            for i, agent in enumerate(env.possible_agents):
                root = roots[agent]
                if root.bundles:
                    temperature = 1.0 if round_idx < 96 else 1e-3
                    action, probs = mcts.get_action_probs(root, temperature=temperature)
                    actions[agent] = action

                    encoded_state = mcts.encoder.encode(env.state, root.player)
                    action_feat = mcts.encoder.encode_action(root.bundles, root.player)
                    mask = mcts.get_action_mask(root.bundles)
                    trajectory.append((encoded_state, action_feat, mask, probs, root.player))
                else:
                    actions[agent] = 0

            env.step(actions)

        winner = env.state.winner
        final_data = []
        for state, action_feat, mask, probs, player in trajectory:
            val = 0.0
            if winner is not None:
                val = 1.0 if winner == player else -1.0
            final_data.append((state, action_feat, mask, probs, val))

        request_queue.put(('trajectory', final_data))

def evaluator_worker(model_path, best_model_path, result_queue):
    """
    Evaluator asynchronously runs 20 games between current model and best model.
    """
    device = torch.device('cpu')
    current_model = PolicyValueNet.load_checkpoint(model_path, device=device)
    current_model.eval()
    best_model = PolicyValueNet.load_checkpoint(best_model_path, device=device)
    best_model.eval()

    config = SearchConfig()
    mcts_current = ParallelMCTS(config)
    mcts_best = ParallelMCTS(config)

    wins = 0
    draws = 0
    num_games = 20

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

                # Simple 1-step evaluation for fast eval
                encoded = mcts.encoder.encode(env.state, i)
                action_feat = mcts.encoder.encode_action(bundles, i)
                mask = mcts.get_action_mask(bundles)

                with torch.no_grad():
                    s = torch.tensor(encoded, dtype=torch.float32, device=device).unsqueeze(0)
                    a = torch.tensor(action_feat, dtype=torch.float32, device=device).unsqueeze(0)
                    m = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
                    policy, _ = model(s, a, m)
                    policy = policy[0].numpy()

                action = int(np.argmax(policy[:len(bundles)]))
                actions[agent] = action

            env.step(actions)

        if env.state.winner == current_side:
            wins += 1
        elif env.state.winner is None:
            draws += 1

    win_rate = wins / num_games
    result_queue.put(win_rate)

def train_pipeline():
    mp.set_start_method('spawn', force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs('checkpoints', exist_ok=True)
    model = PolicyValueNet()
    model.to(device)
    model.share_memory()

    # Save initial best model
    os.system("cp checkpoints/latest_model.pt checkpoints/best_model.pt")
    model.save_checkpoint('checkpoints/latest_model.pt')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=50000)

    num_workers = min(4, os.cpu_count() or 1)
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
    batch_states, batch_action_feats, batch_masks, batch_worker_ids = [], [], [], []

    # Throttle training to trajectory collection
    trajectories_collected = 0
    last_trajectories_trained = 0

    try:
        while True:
            # Aggregate evaluations
            while not request_queue.empty() and len(batch_states) < 32:
                try:
                    msg = request_queue.get_nowait()
                    if msg[0] == 'trajectory':
                        trajectories_collected += 1
                        for data in msg[1]:
                            buffer.push(*data)
                    elif msg[0] == 'eval':
                        _, worker_id, state_np, action_feat_np, mask_np = msg
                        for s, a, m in zip(state_np, action_feat_np, mask_np):
                            batch_states.append(s)
                            batch_action_feats.append(a)
                            batch_masks.append(m)
                            batch_worker_ids.append(worker_id)
                except Empty:
                    break

            if batch_states:
                with torch.no_grad():
                    s = torch.tensor(np.stack(batch_states), dtype=torch.float32, device=device)
                    a = torch.tensor(np.stack(batch_action_feats), dtype=torch.float32, device=device)
                    m = torch.tensor(np.stack(batch_masks), dtype=torch.float32, device=device)
                    policies, values = model(s, a, m)
                    policies = policies.cpu().numpy()
                    values = values.cpu().numpy()

                # Group responses by worker
                responses = {w_id: ([], []) for w_id in set(batch_worker_ids)}
                for i, worker_id in enumerate(batch_worker_ids):
                    p_list, v_list = responses[worker_id]
                    p_list.append(policies[i])
                    v_list.append(values[i])

                for worker_id, (p_list, v_list) in responses.items():
                    response_queues[worker_id].put((np.stack(p_list), np.stack(v_list)))

                batch_states.clear()
                batch_action_feats.clear()
                batch_masks.clear()
                batch_worker_ids.clear()

            # Train if we have enough data and respect a sample/update ratio
            if len(buffer) >= 2048 and (trajectories_collected - last_trajectories_trained) > 0:
                updates = trajectories_collected - last_trajectories_trained
                last_trajectories_trained = trajectories_collected

                for _ in range(updates):
                    states, action_feats, masks, policies, values = buffer.sample(256)

                    s = torch.tensor(states, dtype=torch.float32, device=device)
                    a = torch.tensor(action_feats, dtype=torch.float32, device=device)
                    m = torch.tensor(masks, dtype=torch.float32, device=device)
                    p_target = torch.tensor(policies, dtype=torch.float32, device=device)
                    v_target = torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(1)

                    optimizer.zero_grad()
                    p_pred, v_pred = model(s, a, m)

                    v_loss = torch.nn.functional.mse_loss(v_pred, v_target)
                    p_loss = -(p_target * torch.log(p_pred + 1e-8)).sum(dim=1).mean()
                    loss = v_loss + p_loss

                    loss.backward()
                    optimizer.step()

                    step += 1

                    if step % 100 == 0:
                        print(f"Step {step}, Value Loss: {v_loss.item():.4f}, Policy Loss: {p_loss.item():.4f}, Buffer: {len(buffer)}")

                    if step % 1000 == 0:
                        model.save_checkpoint('checkpoints/latest_model.pt')
                        if eval_process is None or not eval_process.is_alive():
                            print("Starting evaluation...")
                            eval_process = mp.Process(
                                target=evaluator_worker,
                                args=('checkpoints/latest_model.pt', 'checkpoints/best_model.pt', eval_result_queue)
                            )
                            eval_process.start()

            if not eval_result_queue.empty():
                win_rate = eval_result_queue.get()
                print(f"Evaluation finished! Win rate: {win_rate * 100:.2f}%")
                if win_rate > 0.55:
                    print("New best model! Updating checkpoints/best_model.pt")
                    os.system("cp checkpoints/latest_model.pt checkpoints/best_model.pt")

    except KeyboardInterrupt:
        print("Stopping training...")
    finally:
        for w in workers:
            w.terminate()
        if eval_process is not None:
            eval_process.terminate()

if __name__ == '__main__':
    train_pipeline()
