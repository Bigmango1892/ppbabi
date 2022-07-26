from fuzzywuzzy import fuzz
import xlrd
import random
from tqdm import tqdm
from itertools import combinations
from collections import deque
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def preprocess_word(word):
    word = word.replace('能力', '')
    word = word.replace('处理', '')
    word = word.replace('功底', '')
    return word

# 读入数据
# workbook = xlrd.open_workbook(r'../0627测试结果.xls','r')
# worksheet = workbook.sheet_by_name("Sheet1")
# col_values = worksheet.col_values(2)
# pic = open(r'jd0.data','rb')
# col_values = pickle.load(pic)

def find_2_similiar_words(values):
    score_dict = {}
    for comb in combinations(values,2):
        comb = list(comb)
        score = fuzz.ratio(preprocess_word(comb[0]),preprocess_word(comb[1]))
        # 需要取阈值，提高阈值可能会使遗漏的变多
        if score > 52:
            score_dict[str(comb[0]+','+str(comb[1]))] = [score]
    # 保存
    # df = pd.DataFrame.from_dict(score_dict).T
    # df.to_csv('similiar_words_ratio_产品经理.csv',index = True)
    # score_dict = sorted(score_dict.items(), key = lambda a : a[1], reverse= True)
    return score_dict

def draw_graph(complete_graph, subgraph_list):
    plt.rcParams['font.sans-serif'] = 'SimHei'
    pos = nx.spring_layout(nx.subgraph(complete_graph, subgraph_list))
    weights = nx.get_edge_attributes(nx.subgraph(complete_graph,subgraph_list), "weight")
    nx.draw_networkx_edge_labels(nx.subgraph(complete_graph,subgraph_list), pos, edge_labels=weights)
    nx.draw_networkx(nx.subgraph(subgraph_list,subgraph_list),pos,font_size= 6)
    plt.show()
# calculate the numbers of edges of a set of nodes
def cal_edges(complete_graph, subgraph_list):
    num_edges = []
    max_subgraph = nx.subgraph(complete_graph, subgraph_list)
    for word in subgraph_list:
        num_edges.append(len([i for i in max_subgraph if complete_graph.has_edge(i,word)]))
    print(num_edges)

def merge_cliques(complete_graph):
    merge_result = []
    for subgraph in nx.clique.find_cliques(complete_graph):
        if merge_result == []:
            merge_result.append(subgraph)
            continue
        flag = 0
        for i in merge_result:
            jiao = len(set(i) & set(subgraph))
            if 1.0* jiao / (max(len(i),len(subgraph)))>= 0.5:
                merge_result.remove(i)
                merge_result.append(list(set(i).union(set(subgraph))))
                flag = 1
                break
        if flag == 0: merge_result.append(subgraph)
    for subset in merge_result:
        print(subset)

# 对每一张超过节点数量限制的子图都要进行分解
# return: result_graphs: list[list[node(str)]]
# tmp_graphs: deque(Graph)
def decompose_graph(graph, max_nodes = 15):
    result_graphs = []
    tmp_graphs = deque()
    tmp_graphs.append(graph)
    while len(tmp_graphs) > 0:
        l = len(tmp_graphs)
        for i in range(l):
            cur_graph = tmp_graphs.popleft()
            weights = nx.get_edge_attributes(cur_graph, "weight")
            min_weight = min(weights.values())
            min_edges = [edge for edge in weights.keys() if weights[edge] == min_weight]
            for node1, node2 in min_edges:
                cur_graph.remove_edge(node1,node2)
            for sub_graph in nx.connected_components(cur_graph):
                if len(sub_graph) <= 15:
                    result_graphs.append(sub_graph)
                else:
                    tmp_graphs.append(nx.Graph(nx.subgraph(cur_graph, sub_graph)))
    result_graphs_small = []
    for graph in result_graphs:
        if len(graph)<=3:
            result_graphs.remove(graph)
            result_graphs_small.append(list(graph))
    result_graphs_small = list(sum(result_graphs_small,[]))
    result_graphs_small = find_2_similiar_words(result_graphs_small)
    word_list = list(set(sum([word.split(',') for word in result_graphs_small.keys()], [])))
    edges = [list(word.split(',')) for word in result_graphs_small.keys()]
    word_graph = nx.Graph(nodes=word_list)
    word_graph.add_edges_from(edges)
    for sub_graph in nx.connected_components(word_graph):
        result_graphs.append(sub_graph)
    return result_graphs

def decompose_graph_all(complete_graph, max_nodes = 15):
    result_graphs,tmp_graphs = [], deque()
    for sub_graph in nx.connected_components(complete_graph):
        if len(sub_graph) < max_nodes:
            result_graphs.append(list(sub_graph))
        else:
            tmp_graphs.append(nx.Graph(nx.subgraph(complete_graph, sub_graph)))
    while len(tmp_graphs) > 0:
        l = len(tmp_graphs)
        for i in range(l):
            cur_graph = tmp_graphs.popleft()
            weights = nx.get_edge_attributes(cur_graph, "weight")
            min_weight = min(weights.values())
            min_edges = [edge for edge in weights.keys() if weights[edge] == min_weight]
            for node1, node2 in min_edges:
                cur_graph.remove_edge(node1,node2)
            for sub_graph in nx.connected_components(cur_graph):
                if len(sub_graph) <= 15:
                    result_graphs.append(sub_graph)
                else:
                    tmp_graphs.append(nx.Graph(nx.subgraph(cur_graph, sub_graph)))
    result_graphs_small = []
    for graph in result_graphs:
        if len(graph)<=3:
            result_graphs.remove(graph)
            result_graphs_small.append(list(graph))
    result_graphs_small = list(sum(result_graphs_small,[]))
    result_graphs_small = find_2_similiar_words(result_graphs_small)
    word_list = list(set(sum([word.split(',') for word in result_graphs_small.keys()], [])))
    edges = [list(word.split(',')) for word in result_graphs_small.keys()]
    word_graph = nx.Graph(nodes=word_list)
    word_graph.add_edges_from(edges)
    for sub_graph in nx.connected_components(word_graph):
        result_graphs.append(sub_graph)
    for item in result_graphs:
        print(item)
    return result_graphs

def graph_related(edge_dict, remove_max = True):
    word_list = list(set(sum([word.split(',') for word in edge_dict.keys()],[])))
    edges = [list(word.split(',')) for word in edge_dict.keys()]
    weighted_edges = []
    for i in range(len(edges)):
        edges[i].append(edge_dict[','.join(edges[i])][0])
        weighted_edges.append(edges[i])

    word_graph = nx.Graph(nodes = word_list)
    word_graph.add_weighted_edges_from(weighted_edges)

    result,max_subgraph_list = [],[]
    for sub_graph in nx.connected_components(word_graph):
        result.append(list(sub_graph))
    if remove_max:
        max_subgraph_list = list(max(nx.connected_components(word_graph), key=len))
        result.remove(max_subgraph_list)
        max_subgraph = nx.subgraph(word_graph, max_subgraph_list)
        for node1, node2 in max_subgraph.edges():
            word_graph.remove_edge(node1,node2)
        return result, word_graph
    else:
        result_graphs = decompose_graph_all(word_graph)
        return result_graphs
        # word_graph.remove_edge(node1, node2)
        # if len(sub_graph) < 15:
        #     result.append(sub_graph)
        # else:
        #     max_subgraph_list.append(sub_graph)
    # return result, word_graph

    # 边的数量不是很好的衡量依据，少边的节点，相似度也会有很高的出现。那我能不能砍边呢？
    # 如何砍边：不能直接去找边多的节点，它可能就是边很多。直接去砍权重最低的边
    # for item in max_subgraph_list:
    #     decompose_results = decompose_graph(nx.Graph(nx.subgraph(word_graph, item)))
    #     for item in decompose_results:
    #         print(item)
    # for item in result:
    #     print(item)
    # max_subgraph = nx.subgraph(word_graph, max_subgraph_list)
    # # weights: dict{edge:weight}
    # weights = nx.get_edge_attributes(nx.subgraph(word_graph, max_subgraph_list), "weight")
    # unfrozen_graph = nx.Graph(max_subgraph)
    # for i in range(5):
    #     weights = nx.get_edge_attributes(unfrozen_graph, "weight")
    #     weight = min(weights.values())
    #     min_edges = [edge for edge in weights.keys() if weights[edge] == weight]
    #     for node1, node2 in min_edges:
    #         unfrozen_graph.remove_edge(node1,node2)
    # for sub_graph in nx.connected_components(unfrozen_graph):
    #     print(sub_graph)

def list_intersection(list1, list2):
    list1 =[set(list1[i]) for i in range(len(list1))]
    list2 = [set(list2[i]) for i in range(len(list2))]
    intersection = []
    [list1, list2] = [list1, list2] if len(list1) <= len(list2) else [list2, list1]
    for e1 in list1:
        if e1 in list2: intersection.append(e1)
    return intersection

def graph_union(g1, g2):
   for node1, node2 in g2.edges:
       g1.add_edge(node1, node2)
   return g1

def monte(values, iter_times):
    l = len(values)
    print(l)
    result = []
    score_dict = find_2_similiar_words(values[:int(0.8* l)])
    _, result_graph = graph_related(score_dict)
    pbar = tqdm(total = iter_times)
    for i in range(iter_times):
        pbar.update(1)
        random.shuffle(values)
        score_dict = find_2_similiar_words(values[:int(0.8* l)])
        cur_result_list,cur_result_graph = graph_related(score_dict)
        result_graph = graph_union(result_graph,cur_result_graph)
        max_subgraph_list = list(max(nx.connected_components(result_graph), key=len))
        max_subgraph = nx.subgraph(result_graph, max_subgraph_list)
        for node1, node2 in max_subgraph.edges():
            result_graph.remove_edge(node1, node2)
    for sub_graph in nx.connected_components(result_graph):
        print(sub_graph)
        result.append(list(sub_graph))
    # print(result)
    result = sum([list(e) for e in result],[])
    print(len(result))


if __name__ == "__main__":
    pic = open(r'jd0.data', 'rb')
    col_values = pickle.load(pic)
    col_values = list(set(sum(col_values, [])))
    monte(col_values,160)
    # score_dict = find_2_similiar_words(col_values)
    # result = graph_related(score_dict,False)

    # for i in range(1,17):
    #     pic = open(r'../test/test{}.data'.format(''.join(str(i))), 'rb')
    #     col_values = pickle.load(pic)
    #     col_values = list(set(col_values))
    #     score_dict = find_2_similiar_words(col_values)
    #     result = graph_related(score_dict)
    #     result = np.array(result)
    #     print(result)
    #     np.save('../result/result{}.npy'.format(str(i)),result)