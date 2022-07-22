import time
import numpy as np

import matplotlib.pyplot as plt
from collections import defaultdict

from numpy import true_divide

# Python Program for Floyd Warshall Algorihm

# Import function to initialize the dictionary


V = 0

E = 0

INF = 99999
MAX_INT = INF

# Solves all pair shortest path
# via Floyd Warshall Algorithm

def floydWarshall(graphss):
    for i in range(V):
        for j in range(E):
            if(graphss[i][j]==0):
                graphss[i][j]=INF

    dist = list(map(lambda i: list(map(lambda j: j, i)), graphss))
    for j in range(V):
        for i in range(V):
            for k in range(V):
                dist[i][k] = min(dist[i][j],dist[i][j] + dist[j][k])
    printSolution(dist)

# A utility function to print the solution
def printSolution(dist):
    for i in range(V):
        for j in range(V):
            if(dist[i][j] == INF):
                with open("flyodoutput1.txt", "a") as o:
                    o.write("INF"+' ')
            else:
                with open("flyodoutput1.txt", "a") as o:
                    o.write(str(dist[i][j])+' ')
            if j == V-1:
                with open("flyodoutput1.txt", "a") as o:
                    o.write('\n')
    with open("flyodoutput1.txt", "a") as o:
        o.write('\n')
        o.write('\n')  
        o.write('\n')                

# Dijkstra Algorithm for Modified
def Dijkstra(graph, reweighted_graph, src):

    # Number of vertices in the graph
    num_vertices = len(graph)
    visited = defaultdict(lambda : False)

    # Shortest distance of all vertices from the source
    dist = [INF] * num_vertices
    #print(ans)
    #print("\n")
    #print(sptSet[0])
    dist[src] = 0
    #print(dist)
    for k in range(num_vertices):
        min=MAX_INT
        min_vertex=0
        for i in range(len(dist)):
            if min > dist[i]:
                if visited[i] == False:
                    min =dist[i]
                    min_vertex = i
        curr=min_vertex
        #print(curVertex)
        visited[curr] = True

        for i in range(num_vertices):
           # print(vertex)
           # print(curVertex)
           # print(modifiedGraph)
           # print(sptSet[vertex])
           # print(graph)
            if (dist[i] > (dist[curr] + reweighted_graph[curr][i])):             
                if((visited[i] == False)):   
                    if(graph[curr][i] != 0):
                        dist[i] = (dist[curr] +reweighted_graph[curr][i])
            with open("Johnsonoutput1.txt", "a") as o:
                o.write('Vertex ' + str(i) + ': ' + str(dist[i])+'\n')
    # Print the Shortest distance from the source
    # print(dist)
    # for vertex in range(num_vertices):
    # print ('Vertex ' + str(vertex) + ': ' + str(dist[vertex]))

# Function to calculate shortest distances from source
# to all other vertices using Bellman-Ford algorithm
def BellmanFord(edges, graph, num_vertices,E):

    # Add a source s and calculate its min
    # distance from every other node
    # print(edges)
    # print(len(edges))
    dist = [MAX_INT] * (E+1)
    dist[num_vertices] = 0
   # print(num_vertices)

    for i in range(num_vertices):
        edges.append([num_vertices, i, 0])
   # print(graph)
    for i in range (num_vertices):
        for (src, des, weight) in edges:
            if((dist[src] != MAX_INT) and (dist[src] + weight < dist[des])):
                dist[des] = dist[src] + weight
    
    for i in range(E):
        (src, des, weight) =edges[i]
        if((dist[src] != MAX_INT) and (dist[src] + weight < dist[des])):
            return 0

    # Don't send the value for the source added
    return dist[0:E]

# Implementation of Johnson's algorithm in Python3
def Johnson(graph):

    edges = []

    # Create a list of edges for Bellman-Ford Algorithm
    for i in range(len(graph)):
        for j in range(len(graph[i])):
                edges.append([i, j, graph[i][j]])

    #for i in range(len(edges)):
     #   for j in range(len(edges[i])):
      #      continue
         #   print(edges[i][j]
    
    # Weights used to modify the original weights
    modifyWeights = BellmanFord(edges, graph, V,E)
    if(modifyWeights == 0):
        print("negative weight cycle")
    else:
       # print(len(modifyWeights))

        reweighted_graph = [[0 for x in range(E)] for y in range(V)]
      # Modify the weights to get rid of negative weights
        for i in range(len(graph)):
            #print(len(graph[i]))
            for j in range(len(graph[i])):
            #print(len(graph[i]))
                if graph[i][j] != 0:

                    reweighted_graph[i][j] = (graph[i][j] +modifyWeights[i] - modifyWeights[j])

        #print ('reweighted Graph: ' + str(reweighted_graph))

    # Run Dijkstra for every vertex as source one by one
        for src in range(len(graph)):
          #  print ('\nShortest Distance with vertex ' +
           #             str(src) + ' as the source:\n')
           #call dijkstra for each source vertex
            Dijkstra(graph, reweighted_graph, src)

#def showResults(graph, dst, pointer):
    
#    if dst == None:
#        print(None)
#        return

#    print("Distances:")
#    for (v, row) in zip(graph.vertices(), dst):
#        print(f"{v}: {row}")

#    print("\nPath Pointers:")
#    for (v, row) in zip(graph.vertices(), pointer):
#        print(f"{v}: {row}")


## This function stores the running times 
def graphrepresent(graph):
    start = time.time()
    graphs=graph
    graphss=graph
    floydWarshall(graphss)
    x_floydWarshall.append(E)
    y_floydWarshall.append(round(time.time() - start, 6))
    start = time.time()
    Johnson(graphs)
    x_Johnson.append(E)
    y_Johnson.append(round(time.time() - start, 6))  

x_floydWarshall = []
y_floydWarshall = []
x_Johnson =[]
y_Johnson=[]
graph = []
#intializing the list with zeros
graph=[[0 for x in range(len(graph))] for y in range(len(graph))]
N = 0
M = 0
edges = []
test_cases=-1
flag=1
count=0
K=0
with open("sample1.txt", 'r') as file:
    for line in file:
        test_cases=int(line[0])
        break
    for line in file:
        if K==0:
            count+=1
            if(count>1):
                graphrepresent(graph)
            print(line)
            V, E = line.split()
            V = int(V)
            E = int(E)
            graph=[[0 for x in range(E)] for y in range(V)]
            print('{0} {1} '.format( int(V), int(E)))
            K=int(E)
        elif(K>0):
            l,r,s = line.split(' ')
            graph[int(l)-1][int(r)-1]=int(s)-1
            K=K-1
        #print(K)
    graphrepresent(graph)

#graphs=graph
#floydWarshall(graph)
#floydWarshall(graph)
#Johnson(graphs)

# plot1=plt.bar(x_Johnson, y_Johnson, marker="o")
# plot2=plt.bar(x_floydWarshall, y_floydWarshall, marker="o")
barWidth = 0.20
fig = plt.subplots(figsize =(10, 6))
br1 = np.arange(len(y_Johnson))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
plt.bar(br1,y_floydWarshall, color ='r', width = barWidth,
        edgecolor ='grey', label ='Flyod Warshall')
plt.bar(br2, y_Johnson, color ='g', width = barWidth,
        edgecolor ='grey', label ='Johnson')
plt.legend(["Johnson","Flyod Warshall"])
plt.xlabel("Size")
plt.ylabel("Time")
plt.show()