# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:54:38 2016

@author: john
"""

import networkx as nx
import matplotlib.pyplot as plt
import pydotplus 
from PIL import Image as img

import random 
import math 
import statistics 
import numpy as np

from scipy.interpolate import interp1d


MAX_DEPTH = 8

def find_least_path(g, source, dest): 
    table = [] 
    for i in range(len(g)): 
        table.append(math.inf) 
    table[source] = 0 
    visited = list(range(len(g))) 
    cursor = source 
    while(True): 
        if(cursor == dest): 
            return table[cursor] 
        for n in g[cursor].v_out: 
            table[n] = table[cursor] + 1 
        visited.remove(cursor) 
        least_idx = visited[0] 
        for i in visited[1:]: 
            if(table[i] < table[least_idx]): 
                least_idx = i 
        cursor = least_idx 
    return math.nan


class Node: 
    
    def __init__(self, g, parent, idx): 
        self.graph = g
        self.v_in = []
        self.v_out = []
        self.connects = False 
        self.depth = 0
        self.index = 0
        if(parent is not None):
            self.v_in.append(parent)
            self.depth = self.graph[parent].depth + 1 
            self.index = len(g) 
            self.graph[parent].add_out(idx) 
        
    def connect(self): 
        if(not self.connects): 
            self.connects = True 
            for v in self.v_in: 
                self.graph[v].connect() 

    def add_in(self, v): 
        if(v not in self.v_in): 
            self.v_in.append(v) 
            if(self.graph[v].depth + 1 < self.depth): 
                self.depth = self.graph[v].depth + 1
            #self.depth = find_least_path(self.graph, 0, self.index)
                
    def add_out(self, v): 
        if(v not in self.v_out): 
            self.v_out.append(v) 
            if(v < len(self.graph)):  
                if(self.graph[v].connects): 
                    self.connect() 
                    
    def isConnected(self): 
        return self.connects
                    
def simulate(g, end): 
    history = [] 
    cursor = 0
    while(True): 
        if cursor == end: 
            return len(history) 
        doors = [] 
        for i in g[cursor].v_out: 
            if not (cursor, i) in history: 
                doors.append(i) 
        if len(doors) == 0: 
            doors = g[cursor].v_out 
        door = random.choice(doors) 
        history.append((cursor, door)) 
        cursor = door

def n_sim(g, end, n): 
    results = [] 
    for i in range(n): 
        results.append(simulate(g, end)) 
    results.sort() 
    return results 

def draw_nx(g): 
    drawg = nx.DiGraph() 
    drawg.add_nodes_from(range(len(g)), color = "red")
    for n in range(len(g)): 
        drawg.add_edges_from(list((n, x) for x in g[n].v_out)) 
        if(g[n].isConnected()): 
            drawg.node[n]["color"] = "blue" 
    nx.draw(drawg, arrows = True) 
    
    gp = nx.nx_pydot.to_pydot(drawg) 
    gp.write_png("foo.png", prog = 'dot') 
    im = img.open("foo.png") 
    im.show()

def draw2_nx(g, end): 
    drawg = nx.DiGraph() 
    drawg.add_nodes_from(range(len(g)), color = "red")
    for n in range(len(g)): 
        drawg.add_edges_from(list((n, x) for x in g[n].v_out)) 
        if(g[n].isConnected()): 
            drawg.node[n]["color"] = "blue" 
    nx.draw(drawg, arrows = True) 
    
    drawg.node[0]["color"] = "red" 
    drawg.node[end]["color"] = "red"
    
    gp = nx.nx_pydot.to_pydot(drawg) 
    gp.write_png("foo.png", prog = 'dot') 
    im = img.open("foo.png") 
    im.show()
                       
                   
def mst_branchable(g): 
    b_list = []
    for n in range(len(g)): 
        if((not g[n].isConnected()) and len(g[n].v_out) < 2): 
            b_list.append(n) 
    return b_list
    
def receivable(g, source): 
    r_list = [] 
    for n in range(len(g)): 
        if(g[n].depth >= g[source].depth and n != source): 
            r_list.append(n) 
    random.shuffle(r_list) 
    #r_list.sort(key = lambda n: g[n].depth)
    #print(list(g[i].depth for i in r_list))
    return r_list 
    
def receivable0(g, source): 
    r_list = [] 
    for n in range(len(g)): 
        #if(g[n].depth >= g[source].depth and n != source): 
        if(n != source): 
            r_list.append(n) 
    random.shuffle(r_list) 
    #r_list.sort(key = lambda n: g[n].depth)
    #print(list(g[i].depth for i in r_list))
    return r_list 
    
def make_graph(depth_max): 
    graph = []
    graph.append(Node(graph, None, 0)) 
    graph.append(Node(graph, 0, 1)) 
    graph.append(Node(graph, 0, 2)) 
    
    depth = 1
    deep_idx = 0
    
    while(depth < depth_max): 
        b_list = mst_branchable(graph) 
        b_from = random.choice(b_list) 
        graph.append(Node(graph, b_from, len(graph))) 
        d = 0 
        for n in graph: 
            if(n.depth > d): 
                d = n.depth 
        depth = d
        
    graph[-1].connect()
    end = graph[-1].index 
    
    #draw2_nx(graph, end)
    
    connected = False 
    while(not connected): 
        b_list = mst_branchable(graph) 
        fr_node = random.choice(b_list) 
        # to_node = random.choice(receivable(graph, fr_node))
        to_node = receivable(graph, fr_node)[0]
        
        path_start = find_least_path(graph, 0, fr_node) 
        path_end = find_least_path(graph, to_node, end) 
        path = path_start + path_end
        #print(path) 
        if(path < depth_max - 1): 
            for i in range(depth_max - path): 
                graph.append(Node(graph, fr_node, len(graph))) 
                fr_node = len(graph) - 1 
    
        graph[fr_node].add_out(to_node) 
        graph[to_node].add_in(fr_node) 
        conn_test = True 
        for n in graph: 
            conn_test = conn_test and n.isConnected() 
        connected = conn_test 
        
    #draw2_nx(graph, end)        
        
    return graph, end 
    
#draw2_nx(*(make_graph(8)))
        
#g1, g1end = make_graph(8)
    
#draw2_nx(g1, g1end)     
long = [] 
short = [] 
avg = [] 
n = [] 
rooms = []
for i in range(3, 15): 
    n.append(i) 
    templ = [] 
    temps = [] 
    tempa = [] 
    tempr = []
    for j in range(20): 
        g, gend = make_graph(i)
        tempr.append(len(g))
        run = n_sim(g, gend, 1000) 
        templ.append(max(run))
        temps.append(min(run)) 
        tempa.append(statistics.mean(run)) 
        
    print(n) 
    long.append(max(templ))
    short.append(min(temps)) 
    avg.append(statistics.mean(tempa))  
    rooms.append(statistics.mean(tempr))

print(str(len(n)) + " : " + str(len(long)))    

xnew = np.linspace(3, n[-1], endpoint = True)
f_l = interp1d(np.array(n), np.array(long), kind = 'cubic')
f_s = interp1d(np.array(n), np.array(short), kind = 'cubic')
f_a = interp1d(n, avg, kind = 'cubic')
#f_r = interp1d(np.array(n), np.array(rooms), kind = 'cubic')
plt.figure(1)     
plt.xlabel("Minimum Path Length") 
plt.ylabel("Path Length")
plt.title("Simulated Results")

    
#plt.yscale('log')
plt.plot(n, long, "ro", xnew, f_l(xnew), 'r-') 
plt.plot(n, avg, "bo", xnew, f_a(xnew), 'b-')  
plt.plot(n, short, "go", xnew, f_s(xnew), 'g-') 
#plt.plot(n, rooms, "co", xnew, f_r(xnew), 'c-') 

plt.figure(2)

plt.subplot(121)
plt.xlabel("Minimum Path Length") 
plt.ylabel("Path Length/Number of Nodes")
plt.title("Simulated Results")

    
#plt.yscale('log')
plt.plot(n, long, "ro", xnew, f_l(xnew), 'r-') 
plt.plot(n, avg, "bo", xnew, f_a(xnew), 'b-')  
plt.plot(n, short, "go", xnew, f_s(xnew), 'g-') 
plt.plot(n, rooms, "co", xnew, f_r(xnew), 'c-') 

plt.subplot(122)
plt.xlabel("Minimum Path Length") 
plt.ylabel("Path Length/Number of Nodes")
plt.title("Simulated Results, Log Scale")

    
plt.yscale('log')
plt.plot(n, long, "ro", xnew, f_l(xnew), 'r-') 
plt.plot(n, avg, "bo", xnew, f_a(xnew), 'b-')  
plt.plot(n, short, "go", xnew, f_s(xnew), 'g-') 
plt.plot(n, rooms, "co", xnew, f_r(xnew), 'c-') 


plt.show()

        