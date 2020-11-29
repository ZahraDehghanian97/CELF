import numpy as np
import time
import scipy.io


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


# this function build n realization of probabilistic graph
def build_probable_matrices(adjacency_matrix, mc, p):
    list_m = []
    for i in range(mc):
        temp = np.array(adjacency_matrix)
        for i in range(num_node):
            indexes = np.nonzero(temp[i])
            for j in indexes[0]:
                temp[i, j] = np.random.uniform(0, 1, 1)[0] < p
        list_m.append(temp)
    return list_m


# this function return out neighbor of node
def get_neighbor(g, node):
    return np.nonzero(g[node])


# this function compute Spread if S set over all graph in list_g
def IC(list_g, S):
    spread = []
    for i in range(len(list_g)):
        g = list_g[i]
        new_active, A = S[:], S[:]
        while new_active:
            new_ones = []
            for node in new_active:
                new_ones += list(get_neighbor(g, node))
            new_active = list(set(new_ones[0]) - set(A))
            A += new_active
        spread.append(len(A))
    return np.mean(spread)


# this function compute mean cost of S set in all realization
# in this implementation cost = sigma over weight of edge of S
def cost(S, list_g):
    temp = []
    for s in S:
        temp.append(adjacency_matrix[s])
    temp = np.array(temp)
    cost = 0
    for g in list_g:
        temp1 = []
        for s in S:
            temp1.append(g[s])
        temp1 = np.array(temp1)
        temp2 = np.multiply(temp1, temp)
        cost += sum(sum(temp2))
    if cost == 0: cost += 1
    return cost / len(list_g)


# this function run "lazy hill climbing" Idea to speed up computing marginal gane
def lazy_hill_climbing(g, unitCost_or_benefitRatio):
    if unitCost_or_benefitRatio:  # unit cost version :
        marg_gain = [IC(g, [node]) for node in range(num_node)]
    else:  # benefit ration version :
        marg_gain = [(IC(g, [node]) / cost([node], g)) for node in range(num_node)]
    Q = sorted(zip(range(num_node), marg_gain), key=lambda x: x[1], reverse=True)
    S = [Q[0][0]]
    SPREAD = Q[0][1]
    Q = Q[1:]
    print("add " + str(Q[0][0]) + " node to S , size S = 1 , Spread = "+str(SPREAD))

    flag = True
    counter_s = 1
    while flag:
        check, node_lookup = False, 0
        while not check:
            node_lookup += 1
            current = Q[0][0]
            if unitCost_or_benefitRatio:  # unit cost version :
                Q[0] = (current, IC(g, S + [current]))
            else:  # benefit ratio version :
                Q[0] = (current, IC(g, S + [current]) / cost(S + [current], g))
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
            check = (Q[0][0] == current)
        c = cost(S + [Q[0][0]], g)
        if 0.3 * (Q[0][1]) > c :
            SPREAD = Q[0][1]
            S.append(Q[0][0])
            Q = Q[1:]
            counter_s += 1
            print("add " + str(Q[0][0]) + " node to S , size S = "+str(counter_s)+" , Spread = "+str(SPREAD)+" , cost = "+str(c))
        else:
            flag = False
    return S, SPREAD


# this function run "cost-effective lazy-forward selection" algorithm
# unitCost_or_benefitRatio is a flag to change marginal gane to have theoretically guarantee
def CELF(g):
    start_time = time.time()
    print("<-----unit cost----->")
    S_unit, SPREAD_unit = lazy_hill_climbing(g, True)  # unit cost obj. func.
    time_unit_cost = [time.time() - start_time]
    print("finish unit-cost part ")
    print("time unit-cost = " + str(time_unit_cost))
    start_time2 = time.time()
    print("<-----benefit Ratio ------>")
    S_benefit, SPREAD_benefit = lazy_hill_climbing(g, False)  # benefit Ratio obj. func.
    time_benefit_ratio = [time.time() - start_time2]
    print("finish benefit-ratio part ")
    print("time benefit-ratio = " + str(time_benefit_ratio))
    if SPREAD_benefit > SPREAD_unit:
        S = S_benefit
    else:
        S = S_unit
    final_time = [time.time() - start_time]
    return S, final_time


# load file
input_file = 'facebook101_princton_weighted.mat'
txt_input_file = 'dataset.txt'
# change_MAT_to_TXT(input_file, txt_input_file)
num_node = 6596
adjacency_matrix = build_matrix(txt_input_file, num_node)
print("read input file and convert to matrix")

# genetate realization
num_realization = 1
list_realization = build_probable_matrices(adjacency_matrix, mc=num_realization, p=0.1)
print("generate " + str(num_realization) + " realization successfully")

# Run algorithms
print("start running CELF...")
S, t = CELF(list_realization)
print("celf output:   " + str(S))
print("run time = " + str(t))

# list_q = [[[0, 1, 1], [1, 0, 1], [1, 1, 0]], [[0, 1, 0], [1, 0, 1], [0, 1, 0]]]
# adjacency_matrix = [[0,0.5,1],[0.5,0,0.25],[1,0.25,0]]
# S = [0, 1]
# print(cost(S, list_q))
