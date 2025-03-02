import numpy as np
from matplotlib import pyplot as plt

def distance(p1,p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2) 

def main():
    # Numbre of cells
    Nx = 400
    Ny = 100

    # constants
    tau = 0.53 
    Nt = 3000 #iterations

    # lattice speeds and weights
    NL = 9
    cxs = np.array([ 0, 0, 1, 1, 1, 0,-1,-1,-1]) # nodes x component
    cys = np.array([ 0, 1, 1, 0,-1,-1,-1, 0, 1]) # nodes y component
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36]) # weights of the lattice

    # initial conditions
    F = np.ones((Ny,Nx,NL)) + 0.01 * np.random.randn(Ny,Nx,NL) # Initial flow for every cell for every node
    F[:,:,3] = 2.3 # Every right node of each cell is positive to have a flow that moves to the right 

    obstacle = np.full((Ny,Nx),False)
    obstacle_c = (Nx//4,Ny//2)
    obstacle_radius = 13

    for y in range(0,Ny):
        for x in range(0,Nx):
            if (distance(obstacle_c,(x,y)) < obstacle_radius):
                obstacle[y][x] = True

    # main loop
    for it in range(Nt):
        
        F[:,-1,[6,7,8]] = F[:,-2,[6,7,8]] #avoid reflection on right boundary
        F[:,0,[2,3,4]] = F[:,1,[2,3,4]] #avoid reflection on left boundary

        for i,cx,cy in zip(range(NL),cxs,cys): # index of node, displacement_x, displacement_y
            F[:,:,i] = np.roll(F[:,:,i], cx, axis=1)
            F[:,:,i] = np.roll(F[:,:,i], cy, axis=0)
        
        boundaryF = F[obstacle, :]
        boundaryF = boundaryF[:,[0,5,6,7,8,1,2,3,4]]

        # fluid variables
        rho = np.sum(F,2) # axis 2 is where we have the flow values
        ux = np.sum(F * cxs, 2) / rho # we have displacements
        uy = np.sum(F * cys, 2) / rho 

        F[obstacle,:] = boundaryF
        ux[obstacle] = 0
        uy[obstacle] = 0

        # collision
        Feq = np.zeros(F.shape)
        for i,cx,cy,w in zip(range(NL),cxs,cys,weights):
            Feq[:,:,i] = w * rho * (1  +  3 * (cx*ux + cy*uy)  +  9 * (cx*ux + cy*uy)**2 / 2  -  3 * (ux**2 + uy**2)/2)

        F = F + -(1/tau) * (F-Feq)

        if (it%50==0):
            #swirl
            dfydx = ux[2:  , 1:-1] - ux[0:-2,1:-1]
            dfxdy = uy[1:-1, 2:  ] - uy[1:-1,0:-2]
            curl = dfydx -dfxdy
            plt.imshow(curl,cmap="bwr")
            # plt.imshow(np.sqrt(ux**2 + uy**2)) #speed
            plt.pause(.01)
            plt.cla()

if __name__=="__main__":
    main()