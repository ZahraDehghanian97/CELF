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
    neighbor = []
    list = g[node]
    for i in range(len(list)):
        if list[i] != 0: neighbor.append(i)
    return neighbor


def IC(list_g, S):
    """
    Input:  graph object, set of seed nodes, propagation probability
            and the number of Monte-Carlo simulations
    Output: average number of nodes influenced by the seed nodes
    """

    # Loop over the Monte-Carlo Simulations
    spread = []
    for i in range(len(list_g)):
        g = list_g[i]
        # Simulate propagation process
        new_active, A = S[:], S[:]
        while new_active:

            # For each newly active node, find its neighbors that become activated
            new_ones = []
            for node in new_active:
                # Determine neighbors that become infected
                new_ones += list(get_neighbor(g, node))

            new_active = list(set(new_ones) - set(A))

            # Add newly activated nodes to the set of activated nodes
            A += new_active

        spread.append(len(A))

    return (np.mean(spread))


def greedy(g, k, p=0.1, mc=10):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    S, spread, timelapse, start_time = [], [], [], time.time()

    # Find k nodes with largest marginal gain
    for o in range(k):

        # Loop over nodes that are not yet in seed set to find biggest marginal gain
        best_spread = 0
        for j in set(range(num_node)) - set(S):

            # Get the spread
            s = IC(g, S + [j])

            # Update the winning node and spread so far
            if s > best_spread:
                best_spread, node = s, j

        # Add the selected node to the seed set
        S.append(node)
        print("find " + str(o + 1) + "nd member of S")
        # Add estimated spread and elapsed time
        spread.append(best_spread)
        timelapse.append(time.time() - start_time)

    return (S, spread, timelapse)


def celf(g, k):
    """
    Input:  graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    """

    # --------------------
    # Find the first node with greedy algorithm
    # --------------------

    # Calculate the first iteration sorted list
    start_time = time.time()
    marg_gain = [IC(g, [node]) for node in range(num_node)]

    # Create the sorted list of nodes and their marginal gain
    Q = sorted(zip(range(num_node), marg_gain), key=lambda x: x[1], reverse=True)
    # Select the first node and remove from candidate list
    S, spread, SPREAD = [Q[0][0]], Q[0][1], [Q[0][1]]
    Q, LOOKUPS, timelapse = Q[1:], [num_node], [time.time() - start_time]
    print("find 1st member of S")
    # --------------------
    # Find the next k-1 nodes using the list-sorting procedure
    # --------------------

    for o in range(k - 1):

        check, node_lookup = False, 0

        while not check:
            # Count the number of times the spread is computed
            node_lookup += 1

            # Recalculate spread of top node
            current = Q[0][0]

            # Evaluate the spread function and store the marginal gain in the list
            Q[0] = (current, IC(g, S + [current]) - spread)

            # Re-sort the list
            Q = sorted(Q, key=lambda x: x[1], reverse=True)

            # Check if previous top node stayed on top after the sort
            check = (Q[0][0] == current)

        # Select the next node
        spread += Q[0][1]
        S.append(Q[0][0])
        SPREAD.append(spread)
        LOOKUPS.append(node_lookup)
        timelapse.append(time.time() - start_time)
        print("find "+str(o+2)+"nd member of S")
        # Remove the selected node from the list
        Q = Q[1:]

    return (S, SPREAD, timelapse, LOOKUPS)

# this function build n realization of probabilistic graph
def build_probable_matrixs(adjacency_matrix,mc, p):
    list_m = []
    for i in range(mc):
        temp = np.array(adjacency_matrix)
        for i in range(num_node):
            indexes = np.nonzero(temp[i])
            for j in indexes[0]:
                temp[i, j] =  np.random.uniform(0, 1, 1)[0]< p
        list_m.append(temp)
    return list_m


input_file = 'facebook101_princton_weighted.mat'
txt_input_file = 'dataset.txt'
# change_MAT_to_TXT(input_file, txt_input_file)
num_node = 6596
adjacency_matrix = build_matrix(txt_input_file, num_node)
print("read input file and convert to matrix")
#genetate realization
num_realization = 5
list_realization = build_probable_matrixs(adjacency_matrix,mc=num_realization , p =0.1)
print("generate "+str(num_realization)+" realization successfully")
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
plt.plot(range(1, len(greedy_output[2]) + 1), greedy_output[2], label="Greedy", color="#FBB4AE")
plt.plot(range(1, len(celf_output[2]) + 1), celf_output[2], label="CELF", color="#B3CDE3")
plt.ylabel('Computation Time (Seconds)')
plt.xlabel('Size of Seed Set')
plt.title('Computation Time')
plt.legend(loc=2)
plt.show()
