from dimod import ConstrainedQuadraticModel, Integer, QuadraticModel, Binary, quicksum
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridCQMSampler
from scipy.stats import norm
import numpy as np
from dimod.serialization.format import Formatter
from random import seed
from random import randint
from datetime import datetime
import re

seed()

"""
M = 8 #num di server
K = 7 #num di switch
N = 8 #num di VM
F = 4 #num di flussi
L = 14 #num di collegamenti
"""
#C = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100] #ES FATTIBILE
#C = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
#C = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


#Cs = [10, 10, 10, 10, 10, 10, 10, 10]
#Cs = [100, 100, 100, 100, 100, 100, 100, 100] #ES FATTIBILE
p_s = []
p_sw = []
x = 0
u = [] #utilizzo cpu della j-esima VM sul server i
d = [] #data rate del flusso f sul link l
#pi_idle = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]

#pi_dyn = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] 

M = int(input("Inserisci numero server: "))
N = M
K = int(input("Inserisci numero switch: "))
L = int(input("Inserisci numero link: "))
F = M/2
if F-int(F) != 0:
    F = int(F)
    F += 1
else:
    F = int(F)
C = [10 for i in range (L)]
Cs = [10 for i in range (M)]
pi_idle = [10 for i in range (M + K)]
pi_dyn = [1 for i in range (M + K)]


adj_node =[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
           [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0]]
"""
adj_node = [[0 for j in range(M + K)] for i in range(M + K)]

i = 0
while(i < L):
    x = randint(0, M + K - 1)
    y = randint(0, M + K - 1)
    if (adj_node[x][y] == 0) and (x != y):
        adj_node[x][y] = 1
        adj_node[y][x] = 1
    else:
        i -= 1
    i += 1
"""
#print(adj_node)

src_dst = [[0, 7],
           [2, 1],
           [4, 3],
           [6, 5]]

""" #ES FATTIBILE
src_dst = [[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]]
"""
"""
num_nodes = 2
src_dst = [[0 for j in range(num_nodes)] for i in range(F)]
list_node_gen = []
i = 0
while(i < F):
    j = 0
    while(j < num_nodes):
        src_dst[i][j] = randint(0, M - 1)
        controllo = 0
        if len(list_node_gen) > 0:
            for elem in list_node_gen:
                if elem == src_dst[i][j]:
                    j -= 1
                    controllo = 1
                    break
        if controllo == 0:
            list_node_gen.append(src_dst[i][j])
        j += 1
    i += 1
"""
print(src_dst)

si = [Binary("s" + str(i)) for i in range(M)]
swk = [Binary("sw" + str(i)) for i in range(K)]

v = [[Binary("v" + str(i) + "-" + str(j)) for j in range(N)] for i in range(M)]  
u_v = np.random.normal(8, 1, (M, N))
u_v = u_v.astype(int)
#u_v = [[5 for j in range(N)] for i in range(M)] #ES FATTIBILE
#u_v = [[20 for j in range(N)] for i in range(M)] #ES NON FATTIBILE

rho = [[[Binary("rho" + str(f) + "-" + str(i) + "-" + str(k)) for k in range(K + M)] for i in range(K + M)] for f in range(F)]

d = np.random.normal(4, 1, (F, L))
d = d.astype(int)
#d = [[5 for j in range(L)] for i in range(F)] #ES FATTIBILE
on = [[Binary("on" + str(i) + "-" + str(j)) for j in range(M + K)] for i in range(M + K)]

cqm = ConstrainedQuadraticModel()

obj1 = quicksum(pi_idle[i] * si[i] for i in range(M))
obj2 = quicksum(pi_dyn[i] * quicksum(u_v[i][j] * v[i][j] for j in range(N)) for i in range(M))
obj3 = quicksum(pi_idle[i] * swk[i - M] for i in range(M, M + K))
obj4 = quicksum(rho[f][i][j] + rho[f][j][i] for i in range(M + K) for j in range (M + K) for f in range(F))

cqm.set_objective(obj1 + obj2 + obj3 + obj4)

#constraints
for i in range(M):
    cqm.add_constraint(quicksum(u_v[i][j] * v[i][j] for j in range(N)) - Cs[i] * si[i] <= 0)

for j in range(N):
    cqm.add_constraint(quicksum(v[i][j] for i in range(M)) == 1)


for f in range(F):
    for i in range(M):
        cqm.add_constraint(quicksum(rho[f][i][k] for k in range(K))  - v[src_dst[f][0]][i] <= 0)

for f in range(F):
    for i in range(M):
        cqm.add_constraint(quicksum(rho[f][i][k] for k in range(K)) - v[src_dst[f][1]][i] <= 0)


for f in range(F):
    for i in range(M):
        cqm.add_constraint(v[src_dst[f][0]][i] - v[src_dst[f][1]][i]  - (quicksum(rho[f][i][k] for k in range(K + M)) - quicksum(rho[f][k][i] for k in range(K + M))) == 0)

for k in range(M, M + K):
    for f in range(F):
        cqm.add_constraint(quicksum(rho[f][n][k]  for n in range(M + K)) - quicksum(rho[f][k][n] for n in range(M + K)) == 0)

count = 0
for i in range(M + K):
    for j in range(M + K):
        if adj_node[i][j] == 1 and j > i:
            cqm.add_constraint(quicksum(d[f][count] * (rho[f][i][j] + rho[f][j][i]) for f in range(F)) - C[count] * on[i][j] <= 0)
            count += 1
count = 0
for j in range(M + K):
    for i in range(M + K):
        if adj_node[i][j] == 1 and i > j:
            cqm.add_constraint(quicksum(d[f][count] * (rho[f][i][j] + rho[f][j][i]) for f in range(F)) - C[count] * on[i][j] <= 0)
            count += 1


#on_node = [sum(on[i]) for i in range(M + K)]

for i in range(M + K):
    for j in range(M + K):
        if adj_node[i][j] == 1:
            if i < M:
                cqm.add_constraint(on[i][j] - si[i] <= 0)
            else:
                cqm.add_constraint(on[i][j] - swk[i - M] <= 0)
            if j < M:
                cqm.add_constraint(on[i][j] - si[j] <= 0)
            else:
                cqm.add_constraint(on[i][j] - swk[j - M] <= 0)


import time
start_time = time.time()
sampler = LeapHybridCQMSampler()
res = sampler.sample_cqm(cqm, label='hpc-project')
#print(res)


#print("fattibili")
feasible_sampleset = res.filter(lambda d: d.is_feasible)
#print(feasible_sampleset)
best_sol = feasible_sampleset.first
#print(best_sol)
print("time: %s" %(time.time() - start_time))
dict = best_sol[0]

count = 0
print("Gli indici da 0 a " + str(M - 1) + " sono i server \nGli indici da " + str(M) + " a " + str(M + K - 1) + " sono gli switch")
for i in dict:
    if dict[i] > 0:
        if count == 0 and re.search("on.*", i) is not None:
            print("Collegamenti attivi: ")
            count += 1
        elif count == 1 and re.search("rho.*", i) is not None:
            print("rho[f, [n1, n2]] = 1 se parte del flusso f-esimo va da n1 ad n2")
            count += 1
        elif count == 2 and re.search("s.*", i) is not None:
            print("Switch/server attivi")
            count += 1
        elif count == 3 and re.search("v.*", i) is not None:
            print("v[i, j]: la j-esima macchina virtuale sul i-esimo server")
            count += 1
        print(i) 
print("Energia: " + str(best_sol[1]))
