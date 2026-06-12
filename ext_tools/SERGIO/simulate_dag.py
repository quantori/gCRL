# DAG simulation for community-based GRN

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import random
import pickle

##### Control Panel #####
# Set the parameters for the community-based GRN simulation

# number and size of each communities
communities_sizes = [20, 18, 15, 12, 16] 
n_communities = len(communities_sizes)

# how many target genes?
n_target_genes = 500

# range of TFs for target gene
n_tfs_for_target_gene_range = (n_communities, 2 * n_communities)

# how many edges connecting communities?
n_comm_connecting_edges = 1

# parameter for scale-free network generation
alpha = 0.1
beta = 0.7 
gamma = 0.2

# Kij absolute ranges for edges
within_communities_Kij_range = (2, 5)
between_communities_Kij_range = (2, 5)
target_genes_Kij_range = (2, 5)

# Hij parameter
Hij = 2

# Bi range for master regulator
Bi_range = [1, 5]

##### Create each community #####

#用 NetworkX 的 scale_free_graph 生成无向混合图，再转为有向图 DiGraph,得到5个带环的分区子图
communities = []
for i in range(n_communities):
    tmp = nx.DiGraph(nx.scale_free_graph(n = communities_sizes[i], alpha = alpha, beta = beta, gamma = gamma, seed = i))
    communities.append(tmp)

# for i in range(5):
#     plt.figure(figsize=(5,5))
#     nx.draw_circular(communities[i], with_labels=True, arrows=True, node_color='lightgreen', edge_color='black')
#     plt.title(f"Community {i+1}")
#     plt.show()

# make each community acyclic
for i in range(n_communities):

    # is it already a DAG?
    if not nx.is_directed_acyclic_graph(communities[i]):

        # if not, then we write the network in a tmp file
        nx.write_edgelist(G = communities[i], path='/home/may0e/gCAL-main/simulated_data/tmp.edges', data=False)

        # we use a function to identify the edges to remove
        subprocess.run([
            "python3",
            "/home/may0e/gCAL-main/simulated_data/breaking_cycles_in_noisy_hierarchies/remove_cycle_edges_by_hierarchy.py",
            "-s", "pagerank",
            "-g", "/home/may0e/gCAL-main/simulated_data/tmp.edges"
        ])
        to_remove = np.loadtxt('/home/may0e/gCAL-main/simulated_data/tmp_removed_by_PG.edges', dtype=int)

        # and now we remove the edges
        for j in range(to_remove.shape[0]):
            communities[i].remove_edge(*(to_remove[j,:]))

    # plt.figure(figsize=(5, 5))
    # nx.draw_circular(
    #     communities[i],
    #     with_labels=True,
    #     arrows=True,
    #     node_color='lightblue',
    #     edge_color='gray'
    # )
    # plt.title(f"Community {i+1} (After Cycle Breaking)")
    # plt.show()

###### remove isolated nodes from each community acyclic ##### 删除孤立点，保证社区内连通性
for i in range(n_communities):

    # any node to remove? In case, proceed!
    to_remove = list(nx.isolates(communities[i]))
    if len(to_remove) > 0 :
        communities[i].remove_nodes_from(to_remove)

    # plt.figure(figsize=(5, 5))
    # nx.draw_circular(
    #     communities[i],
    #     with_labels=True,
    #     arrows=True,
    #     node_color='lightblue',
    #     edge_color='gray'
    # )
    # plt.title(f"Community {i+1} (after removing isolates)")
    # plt.show()

##### adding weights Kij and Hij to the edges ##### 给每个边增加权重
for i in range(n_communities):
    for u,v in communities[i].edges():
        tmp = float(random.uniform(*within_communities_Kij_range) * np.sign(random.uniform(-1, 1)))
        communities[i].edges[u, v]['Kij'] = tmp
        communities[i].edges[u, v]['Hij'] = Hij

for comm_id, G in enumerate(communities):
    for n in G.nodes():
        G.nodes[n]['community'] = comm_id

###### merging the communities, if multiple ones ##### 合并所有社区，形成全局 DAG   
if n_communities == 1:
    dag = communities[0]
else :
    dag = nx.disjoint_union(communities[0], communities[1])
    if n_communities > 2:
        for i in range(2, n_communities) :
            dag = nx.disjoint_union(dag, communities[i])

print(dag)
# nx.draw_circular(dag)
# plt.figure(figsize=(8, 8))
# nx.draw(
#     dag,
#     pos=nx.spring_layout(dag, seed=42),
#     with_labels=True,
#     arrows=True,
#     node_color='lightcoral',
#     edge_color='gray',
#     node_size=500
# )
# plt.title("Combined DAG from All Communities")
# plt.show()

##### making sure the communities are connected ##### 在社区之间加入少量边（默认每对社区加 1 条），保证整个图连通但“松散”相连
to_add = sorted(nx.k_edge_augmentation(nx.Graph(dag), n_comm_connecting_edges))
if len(to_add) > 0 :
    for t in to_add:
        tmp = float(random.uniform(*between_communities_Kij_range) * np.sign(random.uniform(-1, 1)))
        dag.add_edge(*t, Kij = tmp, Hij = Hij)

nx.draw_circular(dag)

# ensuring again that dag is acyclic
if not nx.is_directed_acyclic_graph(dag):

    # we write the network in a tmp file
    nx.write_edgelist(G = communities[i], path='/home/may0e/gCAL-main/simulated_data/tmp.edges', data=False)

    # we use a function to identify the edges to remove
    subprocess.run([
        "python3",
        "/home/may0e/gCAL-main/simulated_data/breaking_cycles_in_noisy_hierarchies/remove_cycle_edges_by_hierarchy.py",
        "-s", "pagerank",
        "-g", "/home/may0e/gCAL-main/simulated_data/tmp.edges"
    ])
    to_remove = np.loadtxt('/home/may0e/gCAL-main/simulated_data/tmp_removed_by_PG.edges', dtype=int)

    # and now we remove the edges
    for j in range(to_remove.shape[0]):
        dag.remove_edge(*(to_remove[j,:]))
nx.draw_kamada_kawai(dag)

##### looping over the tfs #####
# 给每个转录因子（TF）节点分配一个基线表达速率 Bi，在模拟基因调控网络时，
# 需要区分“主调控因子”（可以自主启动表达）和“受控基因”，主调控因子的 Bi（basal rate）提供了系统的输入驱动力，
# 而其他基因只有在接收到调控信号（由 Kij × 上游活性 + Hij 形式的调控函数）后才会被表达
n_tfs = len(list(dag.nodes))
for i in range(n_tfs):

    # if the node is not regulate, add meaningful Bi. Otherwise set it to zero
    if dag.in_degree(i) == 0 :
        dag.nodes[i]['Bi'] = random.uniform(*Bi_range)
    else :
        dag.nodes[i]['Bi'] = 0    

##### ---------------以上都是生成TF/调控基因的DAG------------------ #####

# “靶基因”（target gene）在基因调控网络（GRN）中指的是那些被上游转录因子（TF）直接调控表达的基因

##### adding the target nodes and their connections #####
for n in range(n_tfs, (n_tfs + n_target_genes)) :

    # adding the node
    dag.add_node(n, Bi = 0)
    # print('n : ' + str(n) + ' out of ' + str((n_tfs + n_target_genes)))

    # selecting the TFs
    n_tfs_for_target_gene = random.randint(*n_tfs_for_target_gene_range)
    selected_tfs = random.sample(k = n_tfs_for_target_gene, population = list(range(n_tfs)))

    # adding the edges from tfs to the target gene
    for t in selected_tfs :
        tmp = float(random.uniform(*target_genes_Kij_range) * np.sign(random.uniform(-1, 1)))
        dag.add_edge(t, n, Kij = tmp, Hij = Hij)

##### Save Files #####

# 把当前内存中的完整 DAG 网络给保存下来
# saving
file = open('/home/may0e/gCAL-main/simulated_data/data/dag_networkx.pickle', 'wb')
pickle.dump(dag, file)
file.close()
print(dag)

# 把所有的“主调控基因”信息导出到一个文本文件
# writing the master regulator file
# with open('/home/may0e/gCAL-main/simulated_data/data/dag_master_regulators.txt', 'w') as f:
#     for n in dag.nodes() :
#         if dag.nodes[n]['Bi'] > 0 :
#             to_print = str(int(n)) + ',' + str(dag.nodes[n]['Bi']) + '\n'
#             f.write(to_print)
# 把所有的“主调控基因”信息导出到一个文本文件
with open('/home/may0e/gCAL-main/simulated_data/data/dag_master_regulators.txt', 'w') as f:
    # 写一个 header（可选）
    f.write("TF_id,community,Bi\n")
    for n, d in dag.nodes(data=True):
        if d.get('Bi', 0) > 0:
            comm = d.get('community', -1)
            bi   = d['Bi']
            f.write(f"{n},{comm},{bi}\n")


# 把网络中所有被调控的基因及其调控关系详细地导出成一个文本文件
# 每一行记录一个“被调控节点”的完整交互信息
# writing the interaction file
with open('/home/may0e/gCAL-main/simulated_data/data/dag_interactions.txt', 'w') as f:

    # looping over nodes
    for n in dag.nodes() :

        # is the node regulated by any tf?
        if dag.in_degree(n) > 0 :

            # initializing the different part of the string to write
            selected_tfs = ''
            Kijs = ''
            Hijs = ''

            # filling up the information for each regulator
            for u,v in dag.in_edges(n) :
                selected_tfs = selected_tfs + ',' + str(u)
                Kijs = Kijs + ',' + str(dag.edges()[u,v]['Kij'])
                Hijs = Hijs + ',' + str(dag.edges()[u,v]['Hij'])

            # assemblying and writing the node information
            to_print = str(int(n)) + ',' + str(len(dag.in_edges(n))) + selected_tfs + Kijs + Hijs + '\n'
            f.write(to_print)

# 把整个 DAG 网络里所有基因之间的调控强度打平成一个真实权重矩阵
with open('/home/may0e/gCAL-main/simulated_data/data/dag_networkx.pickle', 'rb') as f:
    dag = pickle.load(f)

n_nodes = dag.number_of_nodes()
node_names = [f'Gene{i+1}' for i in range(n_nodes)]

W_true = pd.DataFrame(0.0, index=node_names, columns=node_names)

for u, v in dag.edges():
    W_true.iloc[u, v] = dag.edges[u, v]['Kij']

W_true.to_csv('/home/may0e/gCAL-main/simulated_data/data/true_W_matrix.csv')
print("The number of n_tfs:",n_tfs)