import numpy as np

def inspect_expert_data(file_path="data/expert_data.npz"):
    print(f"🔍 正在打开文件: {file_path} ...\n")
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"❌ 找不到文件 {file_path}，请检查路径！")
        return

    # 提取数组
    states = data['states']
    masks = data['masks']
    actions = data['actions']
    values = data['values']

    bundle_features = data['bundle_features']
    print(f"Bundle Features: {bundle_features.shape} | 类型: {bundle_features.dtype}")

    N = len(states)
    print(f"✅ 成功加载数据！总样本数 (步数): {N}")

    print("\n📊 [第一关] 数据形状与类型 (Shapes & Dtypes)")
    print("-" * 40)
    print(f"States:  {states.shape} | 类型: {states.dtype} \t(预期: (N, 12, 20, 20), float32)")
    print(f"Masks:   {masks.shape} | 类型: {masks.dtype} \t(预期: (N, 96), bool)")
    print(f"Actions: {actions.shape} | 类型: {actions.dtype} \t(预期: (N,), int64)")
    print(f"Values:  {values.shape} | 类型: {values.dtype} \t(预期: (N,), float32)")

    print("\n📈 [第二关] 值域分布 (Value Ranges)")
    print("-" * 40)
    print(f"Actions 范围: [{actions.min()}, {actions.max()}] \t(预期: 0 到 95 之间)")
    
    win_count = np.sum(values == 1.0)
    loss_count = np.sum(values == -1.0)
    draw_count = np.sum(values == 0.0)
    print(f"Values  分布: 赢 {win_count} 步, 输 {loss_count} 步, 平局/未决 {draw_count} 步")
    
    if np.isnan(states).any():
        print("❌ 警告：States 中存在 NaN (空值)！")

    print("\n🧠 [第三关] 逻辑合法性校验 (Sanity Checks)")
    print("-" * 40)
    # 核心检查：贪心 AI 选的动作，是不是都在 Mask 允许的合法范围内？
    # 巧妙利用 numpy 高级索引：取出每一行中，被选中动作的那个 mask 值
    chosen_mask_values = masks[np.arange(N), actions]
    illegal_moves = np.sum(~chosen_mask_values) # 统计选了 False 的数量
    
    if illegal_moves == 0:
        print("✅ 完美！100% 的动作都落在合法 Mask 范围内。")
    else:
        print(f"❌ 致命错误：发现了 {illegal_moves} 个非法动作！说明收集逻辑有 Bug。")

    print("\n🎲 [第四关] 战术丰富度 (Action Diversity)")
    print("-" * 40)
    unique_actions, counts = np.unique(actions, return_counts=True)
    print(f"贪心 AI 一共使用了 {len(unique_actions)} 种不同的动作选项。")
    
    # 打印最常用的前 5 个动作
    top_indices = np.argsort(counts)[::-1][:5]
    print("最常用的 Top-5 动作索引及频次:")
    for idx in top_indices:
        print(f"  -> 动作索引 {unique_actions[idx]:2d}: 使用了 {counts[idx]} 次 ({counts[idx]/N*100:.1f}%)")
        
    print("\n==========================================")
    print("💡 结论：如果上面没有出现 ❌，你就可以放心地去跑 train_sl.py 了！")

if __name__ == "__main__":
    inspect_expert_data()