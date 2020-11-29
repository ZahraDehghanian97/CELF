import matplotlib.pyplot as plt
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


def get_neighbor(g, node):
    return np.nonzero(g[node])


def IC2(list_g, S):
    score = 0
    for g in list_g:
        for s in S:
            score += np.count_nonzero(g[s])
    score /= len(list_g)
    return score


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


def greedy(g, k):
    S, spread, timelapse, start_time = [], [], [], time.time()
    # Find k nodes with largest marginal gain
    for o in range(k):
        best_spread = 0
        for j in set(range(num_node)) - set(S):
            s = IC2(g, S + [j])
            if s > best_spread:
                best_spread, node = s, j
        S.append(node)
        print("find " + str(o + 1) + "nd member of S")
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)
    return S, spread, timelapse


def celf(g, k):
    start_time = time.time()
    marg_gain = [IC2(g, [node]) for node in range(num_node)]
    Q = sorted(zip(range(num_node), marg_gain), key=lambda x: x[1], reverse=True)
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [num_node], [time.time() - start_time]
    print("find 1st member of S")
    for o in range(k - 1):
        check, node_lookup = False, 0
        while not check:
            node_lookup += 1
            current = Q[0][0]
            Q[0] = (current, IC2(g, S + [current]) - spread)
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
            check = (Q[0][0] == current)
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)
        print("find " + str(o + 2) + "nd member of S")
        Q = Q[1:]
    return S, SPREAD, timelapse, LOOKUPS


# this function build n realization of probabilistic graph
def build_probable_matrixs(adjacency_matrix, mc, p):
    list_m = []
    for i in range(mc):
        temp = np.array(adjacency_matrix)
        for i in range(num_node):
            indexes = np.nonzero(temp[i])
            for j in indexes[0]:
                temp[i, j] = np.random.uniform(0, 1, 1)[0] < p
        list_m.append(temp)
    return list_m


input_file = 'facebook101_princton_weighted.mat'
txt_input_file = 'dataset.txt'
# change_MAT_to_TXT(input_file, txt_input_file)
num_node = 6596
adjacency_matrix = build_matrix(txt_input_file, num_node)
print("read input file and convert to matrix")
# genetate realization
num_realization = 10
list_realization = build_probable_matrixs(adjacency_matrix, mc=num_realization, p=0.1)
print("generate " + str(num_realization) + " realization successfully")
# Run algorithms
print("start running CELF")
celf_output = celf(list_realization, 10)
print("celf output:   " + str(celf_output[0]))
print("start running greedy")
greedy_output = greedy(list_realization, 10)
print("greedy output: " + str(greedy_output[0]))

# Plot settings
plt.rcParams['figure.figsize'] = (9, 6)
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False

# Plot Computation Time
plt.subplots()
plt.plot(range(1, len(greedy_output[2]) + 1), greedy_output[2], label="Greedy", color="#FBB4AE")
plt.plot(range(1, len(celf_output[2]) + 1), celf_output[2], label="CELF", color="#B3CDE3")
plt.ylabel('Computation Time (Seconds)')
plt.xlabel('Size of Seed Set')
plt.title('Computation Time')
plt.legend(loc=2)

# Plot Expected Spread by Seed Set Size
plt.subplots()
plt.plot(range(1, len(greedy_output[1]) + 1), greedy_output[1], label="Greedy", color="#FBB4AE")
plt.plot(range(1, len(celf_output[1]) + 1), celf_output[1], label="CELF", color="#B3CDE3")
plt.xlabel('Size of Seed Set')
plt.ylabel('Expected Spread')
plt.title('Expected Spread')
plt.legend(loc=2)

plt.show()
