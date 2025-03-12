# plot halo from photon_data.csv
# Usage: python plot-halo.py
# Output: halo.png

import numpy as np
import matplotlib.pyplot as plt
import os

data = np.loadtxt('photon_data.csv', skiprows=1, delimiter=',')
E = data[:,0]
T = data[:,3]
# convert theta and phi to screen coordinates
x = np.abs(data[:,1])*180/np.pi*np.cos(data[:,2])
y = np.abs(data[:,1])*180/np.pi*np.sin(data[:,2])
data = np.column_stack((E, x, y, T))

# Plot halo
plt.figure(figsize=(6,6))
# if 5 < E < 7, color = 'red'
plt.scatter(data[(data[:,0]>5) & (data[:,0]<7),1], data[(data[:,0]>5) & (data[:,0]<7),2], s=5.0, color='red', label='5 TeV < E < 7 TeV')
# if 7 < E < 10, color = 'green'
plt.scatter(data[(data[:,0]>7) & (data[:,0]<10),1], data[(data[:,0]>7) & (data[:,0]<10),2], s=5.0, color='green', label='7 TeV < E < 10 TeV')
# if 10 < E < 20, color = 'blue'
plt.scatter(data[(data[:,0]>10) & (data[:,0]<20),1], data[(data[:,0]>10) & (data[:,0]<20),2], s=5.0, color='blue', label='10 TeV < E < 20 TeV')
# show labels
plt.xlabel('degrees')
plt.ylabel('degrees')
plt.legend()
plt.savefig('halo.png', dpi=300)
plt.close()