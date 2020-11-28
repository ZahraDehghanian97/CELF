import random

import scipy.io
import numpy as np


# this function convert MAT file to txt
def change_MAT_to_TXT(input_file, output_file):
    mat = scipy.io.loadmat(input_file)
    A = mat.__getitem__('A')
    file = open(output_file, 'w')
    for i in range(A.shape[0]):
        for j in A[i].nonzero()[1]:
            file.write(str(i) + ' ' + str(j) + ' ' + str(A[i, j]) + '\n')
    file.close()


# this function build adjeceny matrix from txt file
def build_matrix(dataset, size_matrix):
    result = np.zeros([size_matrix, size_matrix])
    A = open(dataset, encoding='utf-8')
    for line in A:
        text = line.split()
        result[int(text[0]), int(text[1])] = abs(float(text[2]))
    return result


def score_add(node, S, A):
    score = 0
    for j in range(len_matrix):
        if A[node, j] > 0 and (j not in S):
            score += 1
    score *= 0.3
    return score


# this function compute cost of choosing a node
def compute_cost(node, adjacency_matrix):
    return sum(adjacency_matrix[node])


# this function run greedy hill climbing with unit cost marginal gane
def greedy_unit_cost(budget, adjacency_matrix):
    print("run unit-cost marginal gane")
    S = []
    cost = 0
    while cost < budget and len(S) < len_matrix:
        print("start finding " + str(len(S) + 1) + 'th member of opt. set with "unit-cost" obj. func.')
        f_si = np.zeros(len_matrix)
        for node in range(len_matrix):
            if node not in S:
                f_si[node] += score_add(node, S, adjacency_matrix)
        score_max = max(f_si)
        node = np.where(f_si == score_max)[0][0]
        cost_node = compute_cost(node, adjacency_matrix)
        if cost + cost_node < budget:
            cost += cost_node
            S.append(node)
            print("add "+str(node)+" to opt. set. remaining budget = "+str(round(budget - cost , 1)))
        else:
            return S
    return S


input_file = 'facebook101_princton_weighted.mat'
txt_input_file = 'dataset.txt'
# change_MAT_to_TXT(input_file, txt_input_file)
adjacency_matrix = build_matrix(txt_input_file, 6596)
len_matrix = len(adjacency_matrix)

# Generate n realization of probabilistic Graph
print("Generate realization ")
n = 10  # number of realization
budget = 1000
S = greedy_unit_cost(budget, adjacency_matrix)
print("S set = " + str(S))
