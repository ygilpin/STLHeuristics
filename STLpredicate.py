#!/usr/bin/env python3

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import copy as cp

class STLpredicate:
    def __init__(self, t1 = 0, t2 = 0, ae = 0, A = 0, b = 0, cdn = 0 , left = 0, right = 0,
            robType = 'pw'):
        self.t1 = t1 # When predicate is active
        self.t2 = t2
        self.A  = A # Equation of a line
        self.b  = b # constant from ^
        self.ae = ae # always or eventually
        self.cdn = cdn # conjuction, disjunction or negation
        self.left = left # other STL predicates for conjuction/disjunction
        self.right = right
        self.robType = robType # Type of robustness: pw is pointwise, t is traditional

    def myRho(self, x, t):
        # Calculate the robustness of myself
        # Should only be called when I have no left or right branch
        rhoVal = 'Nan'
        if self.t1 <= t and t <= self.t2:
            if self.ae == 'a':
                rhoVal = self.b - self.A @ x[:,t]
            else:
                if self.robType == 'pw':
                    # Check if there is satisfaction at any point
                    sat = False
                    for k in range(0, x.shape[1]):
                        if (self.b - self.A @ x[:,k]) > 0:
                            sat = True
                            break
                    deltax = abs(self.b - self.A @ x[:,t])
                    deltat = abs(1 + self.t2 -t)
                    if sat:
                        rhoVal = (deltat)/((deltax)+0.2)
                    else:
                        rhoVal = -(deltax)/((deltat)+0.2)
                else:
                    print('Traditional robustness')
                    rhoVal = self.b - self.A @ x[:,0]
                    for k in range(1, x.shape[0]):
                        pi = self.b - self.A @ x[:,k]
                        if pi > rhoVal:
                            rhoVal = pi
        
        #print("myRho called", rhoVal)
        return rhoVal

    def Rho(self, x, t):
        #print("Robustness function called")
        # Force the calcluation of left and right 
        leftRho = 'Nan'
        rightRho = 'Nan'
        if self.left and self.right:
            leftRho = self.left.Rho(x,t)
            rightRho = self.right.Rho(x,t)
            if leftRho != 'Nan' and rightRho != 'Nan':
                if self.cdn == 'c':
                    return min(leftRho, rightRho)
                elif self.cdn == 'd':
                    return max(leftRho, rightRho)
            elif leftRho  == 'Nan' and rightRho != 'Nan':
                return rightRho
            elif leftRho != 'Nan' and rightRho == 'Nan':
                return leftRho 
        if self.left:
            if self.cdn == 'n':
                return -self.left.Rho(x,t)
            return self.left.Rho(x,t)
        if self.right:
            if self.cdn == 'n':
                return -self.right.Rho(x,t)
            return self.right.Rho(x,t)
        return self.myRho(x,t)

    def  RhoV(self,x):
        traj = self.plant(x)
        p = []
        for t in range(0, x.shape[1]):
            p.append(self.Rho(traj,t))
        return p

    def robustness(self,x):
        return min(self.RhoV(x))

    def robustnessflt(self,xflt):
        # Same robustness, but flattened version
        x = np.reshape(xflt, (2,-1)) 
        return min(self.RhoV(x))

    def __invert__(self):
        return STLpredicate(t1=self.t1, t2=self.t2, cdn='n', left=self, robType = self.robType)

    def __add__(self, other):
        #print("Disjunction between two predicates")
        return STLpredicate(t1=min(self.t1, other.t2), t2=max(self.t2,other.t2), cdn='d',
                left=self, right=other, robType = self.robType)

    def __mul__(self, other):
        #print("Conjunction between two predicates")
        return STLpredicate(t1=min(self.t1, other.t2), t2=max(self.t2,other.t2), cdn='c',
                left=self, right=other, robType = self.robType)

    def rect(t1,t2,ae,x1,x2,y1,y2, robType):
        # First the equations for the lines are needed
        # Top line
        p1 = STLpredicate(t1, t2, ae, np.array([0, 1, 0]), y2, robType=robType)
        # Bottom line
        p2 = STLpredicate(t1,t2,ae,np.array([0, -1, 0]), -y1, robType=robType)
        # Left line
        p3 = STLpredicate(t1,t2,ae,np.array([-1, 0, 0]), -x1, robType=robType) 
        # Right line
        p4 = STLpredicate(t1,t2,ae,np.array([1, 0, 0]), x2, robType=robType)
        return p1 * p2 * p3 * p4

    def cost(self, xflt):
        x = np.reshape(xflt, (2, -1))
        p = self.robustness(x)
        """d = 0
        for t in range(0,x.shape[1] -1):
            deltax = x[:,t] - x[:,t+1] 
            d += np.linalg.norm(deltax)"""
        return - p*(self.t2 - self.t1 + 1)
    
    def bounds(self, x0, y0, length):
        # Returns a tuple for the bound constraints
        # The first point is fixed. 
        bnds = []
        for i in range(1,3):
            if i == 1:
                bnds.append((x0,x0))
            elif i == 2:
                bnds.append((y0, y0))
            for j in range(0, length):
                bnds.append((None,None))
        return bnds 

    def x_guess(self, x0, y0, length):
        # The intial point is x0, y0
        # Random trajectory after that
        x_tail = 9*np.random.rand(2, length)
        x_start = np.array([[x0], [y0]])
        x = np.concatenate((x_start, x_tail), axis=1)
        return x

    def x_rw(self, x0, y0, step, length):
        x = np.ones((1,length))*x0
        y = np.ones((1,length))*y0
        x = np.vstack((x,y))
        for k in range(1, length):
            walk = step*(2*np.random.rand(1,2) - 1)
            x[:,k] = walk + x[:,k-1]
        return x

    
    def plant(self, pos):
        # This plant simply computes the speed
        # First I need to grow the signal space
        x = np.concatenate((pos, np.zeros((1, pos.shape[1]))), axis=0)
        for t in range(1, x.shape[1]):
            deltax = (x[:,t] - x[:,t-1])[0:2] 
            x[2,t] = np.linalg.norm(deltax)
        return x

    def plot3D(self,xmin, xmax, ymin, ymax, points):
        x = np.array([[],[]])
        for i in np.linspace(xmin,xmax,points):
            for j in np.linspace(ymin, ymax, points):
                x = np.concatenate((x,[[i],[j]]),axis=1)

        X = self.plant(x)
        Z = []
        for t in range(0,x.shape[1]):
            Z.append(self.Rho(X,t))

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(X[0,:], X[1,:], Z, c='r', marker='o')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Robustness')
        plt.show()

    def plotsln3D(self, x):
        X = self.plant(x)
        Z = []
        for t in range(0,x.shape[1]):
            Z.append(self.Rho(X,t))

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(X[0,:], X[1,:], Z, c='r', marker='o')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Robustness')
        plt.show()

    def diffEvoBB(self,xinit, yinit, step, length):
        # Black box differential evolution function
        pop_size = 20*length
        n_cross = 5 # Number of forced cross-overs
        cr = 0.8
        max_iter = 3000

        # Generate Initial Population
        pop = []
        score = []
        for i in range(pop_size):
            pop.append(self.x_rw(xinit, yinit, step, length))
            score.append(self.robustness(pop[i]))

        # Evolution 
        converged = False
        n = 0
        pindex = [i for i in range(pop_size)]
        jindex = [i for i in range(1,length)]
        while not converged:
            for i in range(pop_size):
                # Determine Parents Randomly
                parents = np.random.permutation(pindex)
                ii = 0
                if parents[ii] == i:
                    ii +=1
                p1 = parents[ii]
                ii += 1
                if parents[ii] == i:
                    ii +=1
                p2 = parents[ii]
                ii += 1
                if parents[ii] == i:
                    ii +=1
                p3 = parents[ii]
                ii += 1

                # Pick f randomly 
                f = np.random.ranf()*0.5 + 0.5
                """if f < 0.5 or f > 1:
                    print(f)
                    return 0"""
                vc = pop[p1] + f*(pop[p2]-pop[p3])

                # Generate test vector
                tv = cp.deepcopy(pop[i]) #pop[i]
                jindices = np.random.permutation(jindex)
                for j in range(1,length):
                    if j <= n_cross:
                        tv[:,jindices[j]] = vc[:,jindices[j]]
                    elif (np.random.random() > cr):
                        tv[:,j] = vc[:,j]

                # Compare the scores
                tv_score = self.robustness(tv)
                if tv_score > score[i]:
                    score[i] = tv_score
                    pop[i] = tv
                if score[i] > 0.5 or n >= max_iter:
                    converged = True
            print('n: ', n, ' avg: ', np.mean(score), ' max: ', max(score))
            n += 1

        return pop[np.argmax(score)]

    def hillClimbingPW(self,xinit, yinit, step, length):
        # Pointwise differential evolution function
        # That is to say differential evolution is applied
        # to each point individually
        pop_size = 10*length 
        max_iter = 400

        # Generate Initial Population
        pop = []
        score = []
        for i in range(pop_size):
            pop.append(self.x_rw(xinit, yinit, step, length))
            score.append(self.RhoV(pop[i]))
        
        converged = False
        pindex = [i for i in range(pop_size)]
        n = 0
        while not converged:
            for candidate in range(pop_size):
                # Determine Parents Randomly
                parents = np.random.permutation(pindex)
                ii = 0
                if parents[ii] == i:
                    ii +=1
                p1 = parents[ii]
                ii += 1
                if parents[ii] == i:
                    ii +=1
                p2 = parents[ii]
                ii += 1
                if parents[ii] == i:
                    ii +=1
                p3 = parents[ii]
                ii += 1
                print('x: ', ii, 'p1,p2,p3', p1, ',' , p2, ',', p3 , ',')

                # Pick f randomly 
                f = np.random.ranf()*0.5 + 0.5
                for point in range(1,length):
                    # Generate Test Point
                    pd = pop[p1][:,point] + f*(pop[p2][:,point] - pop[p3][:,point])
                    newCandidate = cp.deepcopy(pop[candidate])
                    newCandidate[:,point] = newCandidate[:,point] + pd
                    newCandidate = self.plant(newCandidate)

                    # Compare Scores
                    if self.Rho(newCandidate,point)  > score[candidate][point]:
                        pop[candidate][:,point] = newCandidate[1-2,point]
                        score[candidate] = self.RhoV(pop[p1])

            print('n: ', n, max((min(score[p]) for p in parents)))
            n = n + 1
        if max_iter > n:
            converged = True

        print('n: ', n, max((min(score[p]) for p in parents)))

        return pop[np.argmax(((min(score[p]) for p in parents)))]

if __name__ == '__main__':
    from scipy.optimize import minimize
    # Available Times
    t1 = 0
    t2 = 9
    length = 10 
    step = 4
    robustnessType = 'pw'

    # Some predicates based on these points
    r1 = ~STLpredicate.rect(t1,t2, 'a', 1, 3, 1, 3, robType=robustnessType)
    r2 = STLpredicate.rect(t1,t2, 'e', 6, 9, 6, 9, robType=robustnessType)
    r3 = STLpredicate(t1,t2, 'a', np.array([0,0,1]), 2, robType=robustnessType)
    p = r1*r2*r3
    """guess = p.x_rw(0,0,step,length)
    guess_cmplt = p.plant(guess)
    print(guess_cmplt)"""
    
    # Now time to run optimization
    sln = p.diffEvoBB(0,0,step,length)
    print('Final Robustness: ', p.robustness(sln))
    print('Final cost: ', p.cost(sln))
    print("Solution")
    print(p.plant(sln))
    print("Robustness of predicates 1 (obstacle), 2 (eventually), 3 (speed)")
    print(r1.RhoV(sln))
    print(r2.RhoV(sln))
    print(r3.RhoV(sln))
    print('Combined Robustness Vector')
    print(p.RhoV(sln))
    p.plotsln3D(sln)
