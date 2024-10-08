import matplotlib.pyplot as plt
from matplotlib import collections  as mc
from matplotlib import animation
import numpy as np
import math
import threading
import time

origin_vertices = np.array([[0,0],[17,0],[17,20],[0,20],[0,0]])
lines = []

def get_rotation_matrix(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def rotate_vertices(vertices,theta):
    return (np.matmul(get_rotation_matrix(theta = theta),(vertices.T))).T

def vertices_to_segments(vertices):
    segments = []
    for i in range(len(vertices)-1):
        segments.append([vertices[i],vertices[i+1]])
    return np.array(segments)
    
lines = np.array(lines)

plt.ion()
plt.show()

lc = mc.LineCollection(vertices_to_segments(origin_vertices),colors = [(0.0,0.0,1.0,1.0),(0.0,0.0,1.0,1.0),(1.0,0.0,0.0,1.0),(0.0,0.0,1.0,1.0)])
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.autoscale
ax.set_xlim(-100,300)
ax.set_ylim(-200,300)
ax.set_aspect('equal')

final_translation = np.array([50,50])
final_rotation = math.pi

def update(t,theta):
    vertices = rotate_vertices(origin_vertices,theta)
    vertices = vertices + t
    segments = vertices_to_segments(vertices)
    lc.update({"segments": segments})
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.005)
