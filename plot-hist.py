# plot halo from photon_data.csv
# Usage: python plot-halo.py
# Output: halo.png

import numpy as np
import matplotlib.pyplot as plt
import os

hist_dir = 'hist'
if not os.path.exists(hist_dir):
    os.makedirs(hist_dir)

data = np.loadtxt('data_v1.csv', skiprows=1, delimiter=',')
E = data[:,0]
theta_obs = data[:,1]*180/np.pi
phi_obs = data[:,2]
T = data[:,3]
# # convert theta and phi to screen coordinates
# x = np.abs(data[:,1])*180/np.pi*np.cos(data[:,2])
# y = np.abs(data[:,1])*180/np.pi*np.sin(data[:,2])
# data = np.column_stack((E, x, y, T))

# plot histogram of T
plt.figure(figsize=(6,6))
plt.hist(T, bins=100, color='blue', alpha=0.7)
plt.xlabel('T')
plt.ylabel('Counts')
plt.savefig('T-hist_v1.png', dpi=300)
plt.close()

# plot histogram of theta_obs
plt.figure(figsize=(6,6))
plt.hist(theta_obs, bins=100, color='blue', alpha=0.7)
plt.xlabel('theta_obs')
plt.ylabel('Counts')
plt.savefig('theta-hist_v1.png', dpi=300)
plt.close()

# plot histogram of phi_obs
plt.figure(figsize=(6,6))
plt.hist(phi_obs, bins=100, color='blue', alpha=0.7)
plt.xlabel('phi_obs')
plt.ylabel('Counts')
plt.savefig('phi-hist_v1.png', dpi=300)
plt.close()

# plot histogram of E
plt.figure(figsize=(6,6))
plt.hist(E, bins=100, color='blue', alpha=0.7)
plt.xlabel('E')
plt.ylabel('Counts')
plt.savefig('E-hist_v1.png', dpi=300)
plt.close()