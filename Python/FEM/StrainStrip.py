# Code based on Jorgen Bergstrom 2022
import numpy as np
import math
from matplotlib import pyplot as plt

def shape(xi):
	x,y = tuple(xi)
	N = [(1.0-x)*(1.0-y), (1.0+x)*(1.0-y), (1.0+x)*(1.0+y), (1.0-x)*(1.0+y)]
	return 0.25*np.array(N)

def gradshape(xi):
	x,y = tuple(xi)
	dN = [[-(1.0-y),  (1.0-y), (1.0+y), -(1.0+y)],
		  [-(1.0-x), -(1.0+x), (1.0+x),  (1.0-x)]]
	return 0.25*np.array(dN)

def main():
    # Create mesh
    mesh_el_x = 9
    mesh_el_y = 49
    mesh_le_x = 10
    mesh_le_y = 50
    mesh_nx = mesh_el_x + 1
    mesh_ny = mesh_el_y + 1
    num_nodes    = mesh_nx * mesh_ny
    num_elements = mesh_el_x * mesh_el_y
    mesh_hx      = mesh_le_x / mesh_el_x
    mesh_hy      = mesh_le_y / mesh_el_y
    
    nodes_tmp = []
    for y in np.linspace(0.0, mesh_le_y, mesh_ny):
        for x in np.linspace(0.0, mesh_le_x, mesh_nx):
            nodes_tmp.append([x,y])
    nodes = np.array(nodes_tmp)

    connections = []
    for j in range(mesh_el_y):
        for i in range(mesh_el_x):
            nbase = i + j*mesh_nx
            connections.append([nbase, nbase + 1, nbase + 1 + mesh_nx, nbase + mesh_nx])

    # Material model for plane strain sigma = 2 * mu * epsilon + lambda * epsilon2 * delta # stress = C B U = C * strain matrix
    E = 100.0 #Youngs modulus
    v = 0.48 #Poisson's ratio
    C = E / (1.0+v)/(1.0-2.0*v) * np.array([[1.0-v, v, 0.0],[v,1.0-v,0.0],[0.0,0.0,0.5-v]]) # C stiffnes matrix

    # Create global stiffness matrix K
    K = np.zeros((2*num_nodes, 2*num_nodes))
    q4 = [[x/math.sqrt(3.0),y/math.sqrt(3.0)] for y in [-1.0,1.0] for x in [-1.0,1.0]] # Gaussian quadrature
    B = np.zeros((3,8))
    for c in connections:
        xIe = nodes[c,:]
        Ke = np.zeros((8,8))
        for q in q4:
            dN = gradshape(q)
            J  = np.dot(dN, xIe).T
            dN = np.dot(np.linalg.inv(J), dN)
            B[0,0::2] = dN[0,:]
            B[1,1::2] = dN[1,:]
            B[2,0::2] = dN[1,:]
            B[2,1::2] = dN[0,:]
            Ke += np.dot(np.dot(B.T,C),B) * np.linalg.det(J)
        for i,I in enumerate(c):
            for j,J in enumerate(c):
                K[2*I,2*J]     += Ke[2*i,2*j]
                K[2*I+1,2*J]   += Ke[2*i+1,2*j]
                K[2*I+1,2*J+1] += Ke[2*i+1,2*j+1]
                K[2*I,2*J+1]   += Ke[2*i,2*j+1]
    
    # Nodal forces and boundary conditions
    f = np.zeros((2*num_nodes))
    for i in range(num_nodes):
        if nodes[i,1] == 0.0:
            K[2*i,:]     = 0.0
            K[2*i+1,:]   = 0.0
            K[2*i,2*i]   = 1.0
            K[2*i+1,2*i+1] = 1.0
        if nodes[i,1] == mesh_le_y:
            x = nodes[i,0]
            f[2*i+1] = 20.0
            if x == 0.0 or x == mesh_le_x:
                f[2*i+1] *= 0.5
    
    # solve linea system
    u = np.linalg.solve(K, f)

    # Plot displacement u
    ux = np.reshape(u[0::2], (mesh_ny,mesh_nx))
    uy = np.reshape(u[1::2], (mesh_ny,mesh_nx))
    xvec = []
    yvec = []
    res  = []
    for i in range(mesh_nx):
        for j in range(mesh_ny):
            xvec.append(i*mesh_hx + ux[j,i])
            yvec.append(j*mesh_hy + uy[j,i])
            res.append(uy[j,i])
    t = plt.tricontourf(xvec, yvec, res, levels=14, cmap=plt.cm.jet)
    plt.scatter(xvec, yvec, marker='o', c='b', s=2)
    plt.grid()
    plt.colorbar(t)
    plt.axis('equal')
    plt.show()

if __name__=="__main__":
    main()