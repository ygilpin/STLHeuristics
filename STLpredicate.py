#!/usr/bin/env python3

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import copy as cp
import math 

class STLpredicate:
    def __init__(self, t1 = 0, t2 = 0, ae = 0, A = 0, b = 0, cdn = 0 , left = 0, right = 0, robType = 'pw', minMaxType='n'):
        self.t1 = t1 # When predicate is active
        self.t2 = t2
        self.A  = A # Equation of a line
        self.b  = b # constant from ^
        self.ae = ae # always or eventually
        self.cdn = cdn # conjuction, disjunction or negation
        self.left = left # other STL predicates for conjuction/disjunction
        self.right = right
        self.robType = robType # Type of robustness: pw is pointwise, t is traditional
        self.minMaxType = minMaxType # Type of min/max : 'n' for normal, 'ag', 'sm'
        self.k = 2

    def pmin(self, v):
        if self.minMaxType == 'n': 
            return min(v)
        
        if self.minMaxType == 'ag':
            normfact = max(np.abs(v))
            v = np.divide(v, normfact)
            pos = True
            for element in v:
                if element < 0:
                    pos = False
                    break
            if pos:
                return np.mean(v)
            else:
                return -math.sqrt(abs(np.prod(v)))

        if self.minMaxType == 'el':
            sm = 0
            for element in v:
                sm += math.exp(-self.k*element)
            return -math.log(sm)/self.k
            
    def pmax(self, v):
        return -self.pmin(v)

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
                        rhoVal = (deltat)/((deltax)+0.4)
                    else:
                        rhoVal = -(deltax)/((deltat)+0.4)
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
                    return self.pmin([leftRho, rightRho])
                elif self.cdn == 'd':
                    return self.pmax([leftRho, rightRho])
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

    def RhoV(self,x):
        traj = self.plant(x)
        p = []
        for t in range(0, x.shape[1]):
            p.append(self.Rho(traj,t))
        return p

    def robustness(self,x):
        tmp = self.pmin(self.RhoV(x))
        return tmp

    def robustnessflt(self,xflt):
        # Same robustness, but flattened version
        x = np.reshape(xflt, (2,-1)) 
        return self.pmin(self.RhoV(x))

    def __invert__(self):
        return STLpredicate(t1=self.t1, t2=self.t2, cdn='n', left=self, robType = self.robType,
                minMaxType = self.minMaxType)

    def __add__(self, other):
        #print("Disjunction between two predicates")
        return STLpredicate(t1=min(self.t1, other.t2), t2=max(self.t2,other.t2), cdn='d',
                left=self, right=other, robType = self.robType, minMaxType=self.minMaxType)

    def __mul__(self, other):
        #print("Conjunction between two predicates")
        return STLpredicate(t1=min(self.t1, other.t2), t2=max(self.t2,other.t2), cdn='c',
                left=self, right=other, robType = self.robType, minMaxType=self.minMaxType)

    def rect(t1,t2,ae,x1,x2,y1,y2, robType, minMaxType = 'n'):
        # First the equations for the lines are needed
        # Top line
        p1 = STLpredicate(t1, t2, ae, np.array([0, 1, 0]), y2, robType=robType, minMaxType=minMaxType)
        # Bottom line
        p2 = STLpredicate(t1,t2,ae,np.array([0, -1, 0]), -y1, robType=robType, minMaxType=minMaxType)
        # Left line
        p3 = STLpredicate(t1,t2,ae,np.array([-1, 0, 0]), -x1, robType=robType, minMaxType=minMaxType)
        # Right line
        p4 = STLpredicate(t1,t2,ae,np.array([1, 0, 0]), x2, robType=robType, minMaxType=minMaxType)
        return p1 * p2 * p3 * p4

    def cost(self, xflt):
        x = np.reshape(xflt, (2, -1))
        p = self.robustness(x)
        return - p*(self.t2 - self.t1 + 1)

    def expsum(self,v):
        esum = 0
        for element in v:
            esum += math.exp(-self.k*element)
        return esum

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
        print("Differential Evolution Blackbox")
        # Black box differential evolution function
        pop_size = 20*length
        n_cross = 5 # Number of forced cross-overs
        cr = 0.8
        max_iter = 30

        # Generate Initial Population
        print("Generating initial population")
        pop = []
        score = []
        for i in range(pop_size):
            pop.append(self.x_rw(xinit, yinit, step, length))
            score.append(self.robustness(pop[i]))

        # Evolution 
        print("Evolution Time")
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
        max_iter = 3000 

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
                # The same parents are used for all points
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
                newCandidate = cp.deepcopy(pop[candidate])
                for point in range(1,length):
                    # Generate Test Point
                    pd = pop[p1][:,point] + f*(pop[p2][:,point] - pop[p3][:,point])
                    newCandidate[:,point] = newCandidate[:,point] + pd
                    newCandidateScoreV = self.RhoV(newCandidate)
                    newCandidateScore = self.pmin(newCandidateScoreV)
                    candidateScore = self.pmin(score[candidate])


                    # Compare Scores
                    if newCandidateScoreV[point]  > score[candidate][point] and newCandidateScore > candidateScore:
                        pop[candidate][:,point] = newCandidate[:,point]
                        score[candidate] = newCandidateScoreV
                    else:
                        newCandidate[:,point] = pop[candidate][:,point]

            print('n: ', n, np.mean([min(score[p]) for p in parents]), max((min(score[p]) for p in parents)))
            if max_iter < n:
                converged = True
            n = n + 1

        print('n: ', n, max((min(score[p]) for p in parents)))

        return pop[np.argmax(((min(score[p]) for p in parents)))]

    def WPF(self, xinit, yinit, step, length):
        pop_size = 30*length
        maxiter = 6000 
        
        # Generate Initial Guess 
        print("Generating a good initial guess")
        bCand = self.x_rw(xinit, yinit, step, length) 
        bScore = self.robustness(bCand)

        for i in range(1,pop_size):
            nCand = self.x_rw(xinit, yinit, step, length)
            nScore = self.robustness(nCand)
            if nScore > bScore:
                bCand = nCand
                bScore = nScore
        #print("Best initial guess: ", bScore)
        #print(bCand)

        converged = False
        n = 0
        while maxiter > n:
            # Find the worst point
            pv = self.RhoV(bCand) # robutness vector
            #print('PV: ', pv)
            #print('pv[1:] ', pv[1:])
            imin = np.argmin(pv[1:]) +1 # index of the worst point
            #print(imin)
             
            # Generate the test vector
            bCandPoint = cp.deepcopy(bCand)
            # Generate a potentially better point
            pointC = np.random.rand(1,2) - 0.5
            bCandPoint[:, imin] = bCandPoint[:, imin] + pointC
            pointSV = self.RhoV(bCandPoint)

            while pv[imin] > pointSV[imin]:
                bCandPoint = cp.deepcopy(bCand) # Reset the copy
                pointC = 0.001*np.random.rand(1,2) - 0.0005
                bCandPoint[:, imin] = bCandPoint[:, imin] + pointC
                pointSV = self.RhoV(bCandPoint)
                print('n: ', n, ' Current: ', pv[imin], 'Proposed: ', pointSV[imin])
                n = n + 1

            #print('PV', pv)
            #if self.pmin(pv) < self.pmin(pointSV):
            bCand = cp.deepcopy(bCandPoint)
            #print('n: ', n, ' robustness: ', self.pmin(bCand))

        return bCand

    def TrajOptSS(self, xinit, yinit, step, length):
        # Black box differential evolution function
        # That uses the sum of the robustness instead
        # of the minimum or other approximation at the final step
        pop_size = 20*length
        n_cross = 5 # Number of forced cross-overs
        cr = 0.8
        max_iter = 6000

        # Generate Initial Population
        pop = []
        score = []
        for i in range(pop_size):
            pop.append(self.x_rw(xinit, yinit, step, length))
            score.append(self.expsum(self.RhoV(pop[i])))

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
                tv_score = self.expsum(self.RhoV(tv))
                if tv_score < score[i]:
                    score[i] = tv_score
                    pop[i] = tv
            k = np.argmin(score)
            act = self.robustness(pop[k])
            print('n: ', n, ' avg: ', np.mean(score), ' min: ', score[k], ' Actual: ', act)
            n += 1
            if act > 1 or n >= max_iter:
                converged = True

        return pop[np.argmin(score)]



if __name__ == '__main__':
    from scipy.optimize import minimize
    # Available Times
    t1 = 0
    t2 = 9
    length = 10 
    step = 2
    robustnessType = 'pw'
    mM = 'n'

    # Some predicates based on these points
    r1 = ~STLpredicate.rect(t1,t2, 'a', 1, 3, 1, 3, robType=robustnessType, minMaxType = mM)
    r2 = STLpredicate.rect(t1,t2, 'e', 6, 9, 6, 9, robType=robustnessType, minMaxType = mM)
    r3 = STLpredicate(t1,t2, 'a', np.array([0,0,1]), 2, robType=robustnessType, minMaxType = mM)
    p = r1*r2*r3
    #print(p.pmin([-5, -4]))
    #print(p.pmin([-5, 2]))
    #print(p.pmin([10, 12]))
    #sln = p.x_rw(0,0,4,5)
    #print(sln)
    #print(p.RhoV(sln))
    #print(p.robustness(sln))
    
    # Now time to run optimization
    sln = p.WPF(0,0,step,length)
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
