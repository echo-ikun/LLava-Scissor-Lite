"""LLaVA-Scissor 的核心算法：并查集 + 近似语义连通分量 (SCC)。"""

from collections import defaultdict

import numpy as np


class UnionFind:
    """带路径压缩和按秩合并的并查集 (Union-Find / Disjoint Set)。

    与原始 SCC 实现的行为完全一致。
    """

    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int32)

    def find(self, x: int) -> int:
        """查找根节点（迭代式路径压缩）。"""
        while self.parent[x] != x:
            # 路径压缩：跳父两步，缩短树高
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def batch_union(self, x_arr: np.ndarray, y_arr: np.ndarray) -> None:
        """批量合并操作：将 (x_arr[i], y_arr[i]) 逐一合并。"""
        for x, y in zip(x_arr, y_arr):
            x_root = self.find(int(x))
            y_root = self.find(int(y))
            if x_root == y_root:
                continue

            # 按秩合并：矮树挂高树下
            if self.rank[x_root] < self.rank[y_root]:
                self.parent[x_root] = y_root
            else:
                self.parent[y_root] = x_root
                if self.rank[x_root] == self.rank[y_root]:
                    self.rank[x_root] += 1


def approximate_components(adj_matrix: np.ndarray, epsilon: float = 0.05) -> list[list[int]]:
    """近似语义连通分量 (Semantic Connected Components, SCC)。

    使用随机采样加速：从 n 个节点中采样 O(log(n) / epsilon^2) 个，
    构建稀疏邻接关系后，用并查集进行聚类。
    未采样的孤立节点作为单元素分量返回。

    参数:
        adj_matrix: 布尔邻接矩阵 [n, n]，adj_matrix[i][j] 为 True 表示 i 和 j 相邻。
        epsilon: 近似误差容忍度，控制采样数 = min(n, ceil(log(n) / epsilon^2))。

    返回:
        连通分量列表，每个分量是节点索引的列表，按最小节点索引排序（度数为次关键字）。
        此顺序与原始 llava_arch_zip.py 保持一致。
    """

    n = adj_matrix.shape[0]
    if n == 0:
        return []

    # 标记所有节点为未访问
    all_nodes = np.ones(n)
    all_indices = np.arange(0, n)
    sample_size = min(n, int(np.ceil(np.log(n) / epsilon**2)))
    sampled_nodes = np.random.choice(n, size=sample_size, replace=False)
    all_nodes[sampled_nodes] = 0

    # 为采样节点构建稀疏邻接表
    neighbor_dict = defaultdict(list)
    for i in sampled_nodes:
        neighbors = np.nonzero(adj_matrix[i])[0]
        valid_neighbors = np.intersect1d(neighbors, all_indices, assume_unique=True)
        neighbor_dict[i] = valid_neighbors
        all_nodes[neighbors] = 0  # 标记邻居为已覆盖

    # 未被任何采样节点覆盖的节点作为孤立分量
    remain_nodes = np.nonzero(all_nodes)[0]
    remain_nodes = [[int(element)] for element in remain_nodes]

    # 用并查集批量合并采样节点及其邻居
    uf = UnionFind(n)
    all_x, all_y = [], []
    for i in sampled_nodes:
        for j in neighbor_dict[i]:
            all_x.append(i)
            all_y.append(j)

    if all_x:
        uf.batch_union(np.array(all_x), np.array(all_y))

    # 收集所有连通分量
    sampled_roots = np.array([uf.find(int(i)) for i in sampled_nodes])
    unique_roots = np.unique(sampled_roots)

    components = []
    for root in unique_roots:
        cluster = np.where(uf.parent == root)[0].tolist()
        if cluster:
            components.append([int(node) for node in cluster])
    components.extend(remain_nodes)

    # 按最小节点索引排序（度数为次关键字）
    degrees = np.count_nonzero(adj_matrix, axis=1)

    def sort_key(cluster: list[int]) -> int:
        """排序规则：在簇中选择度数最高的节点；
        若有并列，取索引最小的节点。返回该节点索引。"""
        max_degree = -1
        min_node = float("inf")
        for node in cluster:
            current_degree = degrees[node]
            if current_degree > max_degree or (current_degree == max_degree and node < min_node):
                max_degree = current_degree
                min_node = node
        return int(min_node)

    components.sort(key=sort_key)
    return components
