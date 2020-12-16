import numpy as np
import time


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
    for x in range(mc):
        temp = np.array(adjacency_matrix)
        for i in range(num_node):
            indexes = np.nonzero(temp[i])
            for j in indexes[0]:
                if j >= i :
                    temp[i, j] = np.random.uniform(0, 1, 1)[0] < p
                    temp[j,i] = temp[i,j]
        list_m.append(temp)
    return list_m


# this function return out neighbor of node
def get_neighbor(g, node):
    neighbor = np.nonzero(g[node])
    if not neighbor: return []
    return np.nonzero(g[node])[0]


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
            new_active = list(set(new_ones) - set(A))
            A += new_active
        spread.append(len(A))
    return np.mean(spread)


def IC1(list_g, S):
    score = 0
    for g in list_g:
        neighbor = []
        for s in S:
            neighbor.extend(get_neighbor(g,s))
        score += len(list(set(neighbor)))
    score /= len(list_g)
    return score


# this function compute mean cost of S set in all realization
# in this implementation cost = sigma over weight of edges in S
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
    global save_IC_1
    if unitCost_or_benefitRatio:  # unit cost version :
        marg_gain = [IC1(g, [node]) for node in range(num_node)]
        save_IC_1 = marg_gain
    else:  # benefit ration version :
        marg_gain = [save_IC_1[node] / cost([node], g) for node in range(num_node)]
    Q = sorted(zip(range(num_node), marg_gain), key=lambda x: x[1], reverse=True)
    S , s = [Q[0][0]] , Q[0][1]
    c = cost(S,g)
    if not unitCost_or_benefitRatio :
        SPREAD = c
    else : SPREAD = 1
    SPREAD = SPREAD * Q[0][1]
    print("add " + str(Q[0][0]) + " node to S, Spread = " + str(SPREAD)+" , cost = "+str(c))
    Q = Q[1:]
    flag = True
    counter_s = 1
    while flag:
        check = False
        while not check:
            current = Q[0][0]
            if unitCost_or_benefitRatio:  # unit cost version :
                Q[0] = (current, IC1(g, S + [current])-s)
            else:  # benefit ratio version :
                Q[0] = (current, (IC1(g, S + [current]) / cost(S + [current], g))-s)
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
            check = (Q[0][0] == current)
        c = cost(S + [Q[0][0]], g)
        temp = Q[0][1]
        temp+=s
        if unitCost_or_benefitRatio:
            temp /= c
        if 0.3 * (temp) > 1:
            SPREAD = temp * c
            s+= Q[0][1]
            S.append(Q[0][0])
            counter_s += 1
            print("add " + str(Q[0][0]) + " node to S, spread = "+str(SPREAD)+" , cost = "+str(c))
            Q = Q[1:]
        else:
            NSPREAD = temp * c
            print("not add "+ str(Q[0][0]) + " node to S, spread = "+str(NSPREAD)+" , cost = "+str(c))
            flag = False
    return S, SPREAD


# this function run "cost-effective lazy-forward selection" algorithm
# unitCost_or_benefitRatio is a flag to change marginal gane to have theoretically guarantee
def CELF(g):
    start_time = time.time()
    print("<----- unit cost ----->")
    S_unit, SPREAD_unit = lazy_hill_climbing(g, True)  # unit cost obj. func.
    time_unit_cost = [time.time() - start_time]
    print("finish unit-cost part ")
    print("time unit-cost = " + str(time_unit_cost))
    start_time2 = time.time()
    print("<----- benefit Ratio ------>")
    S_benefit, SPREAD_benefit = lazy_hill_climbing(g, False)  # benefit Ratio obj. func.
    time_benefit_ratio = [time.time() - start_time2]
    print("finish benefit-ratio part ")
    print("time benefit-ratio = " + str(time_benefit_ratio))
    if SPREAD_benefit > SPREAD_unit:
        S = S_benefit
        spread = SPREAD_benefit
    else:
        S = S_unit
        spread = SPREAD_unit
    final_time = [time.time() - start_time]
    return S, spread, final_time


# load file
input_file = 'facebook101_princton_weighted.mat'
txt_input_file = 'dataset.txt'
num_node = 6596
adjacency_matrix = build_matrix(txt_input_file, num_node)
print("read input file and convert to matrix")

# genetate realization
num_realization = 10
list_realization = build_probable_matrices(adjacency_matrix, mc=num_realization, p=0.1)
print("generate " + str(num_realization) + " realization successfully")


# Run algorithms
print("start running CELF...")
save_IC_1 = []
S, spread, t = CELF(list_realization)
print("<----------------result CELF Algorithm ----------------->")
print("celf output =  " + str(S))
print("mean spread value = " + str(spread))
print("run time = " + str(t[0]))


