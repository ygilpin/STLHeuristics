#!/usr/bin/env python3

import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import copy as cp
import math 
import time

class STLpredicate:
    def __init__(self, t1 = 0, t2 = 0, ae = 0, A = 0, b = 0, cdn = 0 , left = 0, right = 0):
        self.t1 = t1 # When predicate is active
        self.t2 = t2
        self.A  = A         # Equation of a line as in Ax < b
        self.b  = b         # constant from ^
        self.ae = ae        # always or eventually
        self.cdn = cdn      # conjuction, disjunction or negation
        self.left = left    # other STL predicates for conjuction/disjunction
        self.right = right
        self.k = 25         # for LSE approximation
        self.alpha = 25     # for exponential fraction approximation
        self.R = 2.5 # Normalization factor
        self.Rp = False # A & b normalized ?
        self.T = 1 # Sample Rate

    # Minimum function (allows for a variety of smooth approximations)
    def pmin(self, v, mMT = 'n'):
        v = np.array(v)
        if mMT == 'n': 
            return min(v)
        
        # Arithmetic Geometric mean attempt; does not work
        #if mMT == 'ag':
        #    N = len(v)
        #    pos = True
        #    for element in v: 
        #        if element <= 0:
        #            pos = False
        #            break
        #    if pos:
        #        v = v + np.ones((1,N))
        #        return math.pow(np.prod(v), 1/N) -1
        #    else:
        #        rsum = 0
        #        for element in v:
        #            if element <= 0:
        #                rsum += element
        #        return rsum/N

        if mMT == 'el' or mMT == 'ef':
            #print(v)
            max_elem = max(v)
            vprime = -self.k*v + self.k*max_elem
            vprime = np.exp(vprime)
            return -math.log(vprime.sum())/self.k + max_elem
            
        if mMT == 'wk':
            num = 0
            denom = 0
            for element in v:
                expnt = math.exp(-self.alpha*element)
                num += element*expnt
                denom += expnt
            return num/denom

    # Maximum function (allows for various approximations)
    def pmax(self, v, mMT = 'n'):
        v = np.array(v)
        if mMT == 'n': 
            return max(v)

        if mMT == 'el':
            max_elem = max(v)
            vprime = self.k*v - self.k*max_elem
            vprime = np.exp(vprime)
            return -math.log(vprime.sum())/self.k + max_elem

        if mMT == 'wk' or mMT == 'ef':
            num = 0
            denom = 0
            for element in v:
                expnt = math.exp(self.alpha*element)
                num += element*expnt
                denom += expnt
            return num/denom
        
        # Once again AGM approximation attempt, but does not work
        #if mMT == 'ag':
        #    N = len(v)
        #    pos = False 
        #    for element in v: 
        #        if element > 0:
        #            pos = True 
        #            break
        #    if pos:
        #        rsum = 0
        #        for element in v:
        #            if element > 0:
        #                rsum += element
        #        return rsum/N
        #    else:
        #        v = np.ones((1,N)) - v
        #        return -math.pow(np.prod(v), 1/N) +1

            
    def myRho(self, x, t, mMT, rbT):
        # Calculate the robustness of myself
        # Should only be called when I have no left or right branch
        rhoVal = 'Nan'
        if 'mMT' == 'ag':
            x = x/self.R
            if ~self.Rp:
                self.A = self.A /self.R
                self.b = self.b/self.R
                self.Rp = True

        if self.t1 <= t and t <= self.t2:
            if self.ae == 'a':
                rhoVal = self.b - self.A @ x[:,t]
            elif self.ae == 'e':
                if rbT == 'pw':
                    # Check if there is satisfaction at any point
                    sat = False
                    for k in range(self.t1, self.t2+1):
                        if (self.b - self.A @ x[:,k]) > 0:
                            sat = True
                            break
                    deltax = abs(self.b - self.A @ x[:,t])
                    deltat = abs(1 + self.t2 -t)
                    if sat:
                        rhoVal = (deltat)/((deltax)+0.4)
                    else:
                        rhoVal = -(deltax)/((deltat)+0.4)
                elif rbT == 'n':
                    #print('Traditional robustness')
                    pi = []
                    for k in range(self.t1, self.t2 +1):
                        pi.append((self.b - self.A @ x[:,k]))
                    rhoVal = self.pmax(pi)
                else: 
                    print('Error invalid robustness type: ' + rbT)
                    return rhoVal
            else:
                print('Error Computing Robustness')
        return rhoVal

    def Rho(self, x, t, mMT, rbT):
        # Works down binary tree to compute robustness
        #print("Robustness function called")
        # Force the calcluation of left and right 
        leftRho = 'Nan'
        rightRho = 'Nan'
        if self.left and self.right:
            leftRho = self.left.Rho(x, t, mMT, rbT)
            rightRho = self.right.Rho(x, t, mMT, rbT)
            if leftRho != 'Nan' and rightRho != 'Nan':
                if self.cdn == 'c':
                    return self.pmin([leftRho, rightRho], mMT)
                elif self.cdn == 'd':
                    return self.pmax([leftRho, rightRho], mMT)
            elif leftRho  == 'Nan' and rightRho != 'Nan':
                return rightRho
            elif leftRho != 'Nan' and rightRho == 'Nan':
                return leftRho 
        if self.left:
            if self.cdn == 'n':
                return -self.left.Rho(x, t, mMT, rbT)
            elif self.cdn == 'e':
                if t <= self.t2 and t >= self.t1:
                    v = []
                    if rbT == 'pw':
                        sat = False
                        for i in range(self.t1,self.t2+1):
                            rho = self.left.Rho(x,i,mMT, rbT)
                            #print(rho)
                            if rho > 0:
                                sat = True
                                break
                        deltax = abs(self.left.Rho(x,t,mMT, rbT))
                        deltat = abs(1 + self.t2 -t)
                        if sat:
                            rhoVal = (deltat)/((deltax)+0.4)
                        else:
                            rhoVal = -(deltax)/((deltat)+0.4)
                        return rhoVal

                    elif rbT == 'n':
                        for i in range(self.t1,self.t2+1):
                            v.append(self.left.Rho(x,i,mMT, rbT))
                        return self.pmax(v)
                else:
                    return 'Nan'
            return self.left.Rho(x, t, mMT, rbT)
        if self.right:
            if self.cdn == 'n':
                return -self.right.Rho(x, t, mMT, rbT)
            elif self.cdn == 'e':
                if t <= self.t2 and t > self.t1:
                    v = []
                    if rbT == 'pw':
                        sat = False
                        for i in range(self.t1,self.t2+1):
                            rho = self.left.Rho(x,i,mMT, rbT)
                            #print(rho)
                            if rho > 0:
                                sat = True
                                break
                        deltax = abs(self.left.Rho(x,t,mMT, rbT))
                        deltat = abs(1 + self.t2 -t)
                        if sat:
                            rhoVal = (deltat)/((deltax)+0.4)
                        else:
                            rhoVal = -(deltax)/((deltat)+0.4)
                        return rhoVal

                    elif rbT == 'n':
                        for i in range(self.t1,self.t2+1):
                            print('Printing self.right')
                            print(self.right)
                            v.append(self.right.Rho(x,i,mMT, rbT))
                        return self.pmax(v)
                else:
                    return 'Nan'
            return self.right.Rho(x, t, mMT, rbT)
        return self.myRho(x, t, mMT, rbT)

    def RhoV(self, x, mMT, rbT):
        # Compute the robustness vector
        traj = self.plant(x)
        p = []
        for t in range(0, x.shape[1]):
            p.append(self.Rho(traj,t, mMT, rbT))
        return p

    def robustness(self,x, mMT, rbT):
        # Compute the robustness from the robustness vector
        tmp = self.pmin(self.RhoV(x, mMT, rbT))
        return tmp

    def robustnessflt(self,xflt, mMT, rbT):
        # Same robustness, but flattened version
        x = np.reshape(xflt, (2,-1)) 
        return self.pmin(self.RhoV(x, mMT, rbT))

    def __invert__(self):
        # allows negation of predicate 
        return STLpredicate(t1=self.t1, t2=self.t2, cdn='n', left=self)

    def __add__(self, other):
        # Allows easy disjuction between two predicates
        #print("Disjunction between two predicates")
        return STLpredicate(t1=min(self.t1, other.t2), t2=max(self.t2,other.t2), cdn='d',
                left=self, right=other)

    def __mul__(self, other):
        # Allows easy disjuction between predicates
        #print("Conjunction between two predicates")
        return STLpredicate(t1=min(self.t1, other.t2), t2=max(self.t2,other.t2), cdn='c',
                left=self, right=other)

    def rect(t1,t2,ae,x1,x2,y1,y2):
        # writes the predicates for inside a rectangle
        # First the equations for the lines are needed
        # Top line
        p1 = STLpredicate(t1, t2, 'a', np.array([0, 1, 0, 0]), y2)
        # Bottom line
        p2 = STLpredicate(t1,t2, 'a', np.array([0, -1, 0, 0]), -y1)
        # Left line
        p3 = STLpredicate(t1,t2, 'a', np.array([-1, 0, 0, 0]), -x1)
        # Right line
        p4 = STLpredicate(t1,t2, 'a', np.array([1, 0, 0, 0]), x2)


        cmplt = p1 * p2 * p3 * p4

        # This compensates for how eventually does not distribute over conjunction
        if ae == 'e':
            return STLpredicate(t1, t2 ,'e', cdn='e', left=cmplt) 
        else:
            return cmplt

    def cost(self, xflt, mMT, rbT):
        # Cost funtion for optimization
        x = np.reshape(xflt, (2, -1))
        x = np.hstack(([[0],[0]],x))
        p = self.robustness(x, mMT, rbT)
        return - p

    def expsum(self,v):
        # Computes the LSE approximation without the log
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
            for j in range(0, length-1):
                bnds.append((None,None))
        return bnds 

    def x_guess(self, x0, y0, length):
        # The intial point is x0, y0
        # Random trajectory after that
        x_tail = 9*np.random.rand(2, length)
        x_start = np.array([[x0], [y0]])
        x = np.concatenate((x_start, x_tail), axis=1)
        return x

    def x_rw(self, v0, omega0, step, length):
        # Initial guess function tuned for TurtleBot Burger
        # This probably the most important function for getting good results
        """v = v0
        theta = theta0
        x = np.zeros((1,length))
        y = np.zeros((1,length))
        for k in range(1, length):
            v += step*(np.random.rand() - 0.5)
            theta += np.pi*(np.random.rand() - 0.5)
            x[0,k] = x[0,k-1] + v*np.cos(theta)
            y[0,k] = y[0,k-1] + v*np.sin(theta)"""
        speed = 0.2*np.ones((1,length)) #0.1*step*(2*np.random.rand(1,length) -1)
        omega = 0.2*(np.random.rand(1,length) -1)
        x = np.vstack((speed,omega))
        x[0,0] = v0
        x[1,0] = omega0
        #return self.WPFT(x, self.RhoV(x, 'ef', 'pw'), 50, 'ef', 'pw')
        return x 

    
    def plant(self, vel):
        # This plant simply computes the x and y position
        # It is currently based on a TurtleBot but the focus is generating the 
        # additional signals in the signal space
        # First I need to grow the signal space
        x = np.concatenate((np.zeros((2, vel.shape[1])), vel), axis=0)
        theta_t = 0
        for t in range(1, x.shape[1]):
            omega = x[3,t]
            x[0,t] = x[0,t-1] + x[2,t]/omega*(np.sin(omega*self.T + theta_t) - np.sin(theta_t))
            x[1,t] = x[1,t-1] + x[2,t]/omega*(np.cos(theta_t) - np.cos(omega*self.T + theta_t))
            theta_t += omega*self.T
        return x

    def plot3D(self,xmin, xmax, ymin, ymax, points):
        # You can think of a predicate as constraining a space
        # So this function plots in 3D the robustness space
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
        print(self.RhoV(x))

    def plotsln3D(self, x, mMT, rbT, s='show'):
        # This function plots a solution 
        # and its robustness in 3D
        X = self.plant(x)
        Z = []
        for t in range(0,x.shape[1]):
            Z.append(self.Rho(X,t, mMT, rbT))

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(X[0,:], X[1,:], Z, c='r', marker='o')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Robustness')
        if s == 'show':
            plt.show()
        else: 
            plt.savefig(s + '.png', dpi=150)

    def diffEvoBB(self,xinit, yinit, step, length, mMT, rbT):
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
            score.append(self.robustness(pop[i], mMT, rbT))

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
                tv_score = self.robustness(tv, mMT, rbT)
                if tv_score > score[i]:
                    score[i] = tv_score
                    pop[i] = tv
                if score[i] > 0.5 or n >= max_iter:
                    converged = True
            print('n: ', n, ' avg: ', np.mean(score), ' max: ', max(score))
            n += 1

        return pop[np.argmax(score)]

    def TrajOptSSDE(self, xinit, yinit, step, length, mMT, rbT):
        # Black box differential evolution function
        # That uses the sum of the robustness instead
        # of the minimum or other approximation at the final step
        pop_size = 20*length
        n_cross = 5 # Number of forced cross-overs
        cr = 0.8
        max_iter = 3000

        # Generate Initial Population
        pop = []
        score = []
        for i in range(pop_size):
            pop.append(self.x_rw(xinit, yinit, step, length))
            score.append(self.expsum(self.RhoV(pop[i], mMT, rbT)))

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
                tv_score = self.expsum(self.RhoV(tv, mMT))
                if tv_score < score[i]:
                    score[i] = tv_score
                    pop[i] = tv
            k = np.argmin(score)
            act = self.robustness(pop[k], mMT, rbT)
            print('n: ', n, ' avg: ', np.mean(score), ' min: ', score[k], ' Actual: ', act)
            n += 1
            if act > 1 or n >= max_iter:
                converged = True

        return pop[np.argmin(score)]
    
    def TrajOptSSmin(self, xinit, yinit, step, length, mMT, rbT):
        # Black box scipy minimize 
        # That uses the sum of the robustness instead
        # of the minimum or other approximation at the final step
        bnds = self.bounds(xinit, yinit, length)
        
        # Generate a good initial guess
        pop_size = 10*length 
        pop = []
        score = []
        for i in range(pop_size):
            pop.append(self.x_rw(xinit, yinit, step, length))
            score.append(self.robustness(pop[i],mMT, rbT))

        elite_i = np.argmax(score)
        print(score[elite_i])
        guess = pop[elite_i]
        guessflt = np.reshape(guess[:,1:], (1, -1))
        options = {'maxiter': 1000}
        # args = (mMT,rbT)
        opt = sciOpt.minimize(self.expsum, guessflt,method='SLSQP',options=options)
        #opt = minimize(self.expsum, guessflt, method='TNC', bounds=bnds)
        print(opt.message)
        return np.reshape(opt.x, (2,-1))

    def VanillaMin(self, xinit, yinit, x0, mMT='ef', rbT='n', mthd='SLSQP'):
        # Generic single valued cost function optimization

        # Depending on your plant you may need to force bounds at particular points
        # See the commented out parts here and the bounds function or some example work
        #bnds = self.bounds(xinit, yinit, length)
        self.k = 10 # For some reason this function likes to overflow so smaller k and alpha
        self.alpha = 10 
        guessflt = np.reshape(x0[:,1:], (1, -1))
        options = {'maxiter': 1000}
        opt = sciOpt.minimize(self.cost, guessflt, args=(mMT,rbT),method=mthd, options=options)
        """bnds = []
        for i in range(1,3):
            for j in range(1, length):
                bnds.append((0,2))
        opt = sciOpt.differential_evolution(self.cost, 
                bounds=bnds, 
                args=(mMT,rbT), 
                maxiter=5000,
                disp=True,
                polish=True,
                workers=-1)
        #print(len((bnds)))
        #print(len(guessflt))
        opt = sciOpt.dual_annealing(self.cost,
                bounds=bnds, 
                args=(mMT,rbT), 
                maxiter=5000,
                x0=guessflt)
        print(opt.message)"""
        sln = np.reshape(opt.x, (2,-1))
        return np.hstack(([[0],[0]],sln))


    # Genetic Evolution 
    def mutate(self, pop, popscore, popsize, elite, eliteScoreV, length):
        for i in range(0, popsize):
            if i == elite:
                pop[elite] = self.WPFT(pop[elite], popscore[elite], 100)
                #print("Not applying mutation to elite: ", elite)
                continue
            for j in range(1, length):
                pop[i][:,j] = pop[i][:,j] + 0.05*abs(10 - popscore[i][j])*(np.random.ranf() -0.5)
                #pop[i][:,j] = pop[i][:,j] + 0.2*(np.random.ranf((1,2)) -0.5)
        return pop

    def mutateAgent(self, agent, popscore, length):
        # The mutation function for the genetic optimization
        for j in range(1, length):
            #agent[:,j] = agent[:,j] + 0.01*abs(10 - popscore[i][j])*(np.random.ranf() -0.5)
            agent[:,j] = agent[:,j] + 0.3*(np.random.ranf((1,2)) -0.5)
        return agent


    def reproduce(self, pop, popscore, p1, p2, length):
        # The function for making children in genetic evolution optimization
        child1 = np.empty((2,length)) 
        child2 = np.empty((2,length)) 
        for index in range(0, length):
            beta = np.random.ranf()*0.5
            child1 = pop[p1]*beta + (1-beta)*pop[p2]
            child2 = pop[p1]*(1-beta) + beta*pop[p2]
            """ 
            x = popscore[p1][index]
            y = popscore[p2][index]
            if x > y:
                if 1 - y/x > np.random.ranf():
                    child[:,index] = pop[p1][:,index]
                else:
                    child[:,index] = pop[p2][:,index]
            else:
                if 1 - x/y > np.random.ranf():
                    child[:, index] = pop[p2][:,index]
                else:
                    child[:,index] = pop[p1][:,index]
            """        
        return [child1, child2]

    def WPFT(self, agent, agentScoreV, n, mMT, rbT):
        # The modified Worst Point First Optimizer for genetic optimization
        j = 0
        agentcp = cp.deepcopy(agent)
        imin = np.argmin(agentScoreV[1:])
        agentcpScoreV = self.RhoV(agentcp, mMT, rbT)
        while agentScoreV[imin] >= agentcpScoreV[imin] and j < n-1:
            agentcp = cp.deepcopy(agent) # Reset the copy
            pointC = 0.1*(np.random.rand(1,2) - 0.5)
            agentcp[:, imin] = agentcp[:, imin] + pointC
            agentcpScoreV = self.RhoV(agentcp, mMT, rbT)
            #print('j: ', j, ' Current: ', agentScoreV[imin], 'Proposed: ', agentcpScoreV[imin])
            j = j + 1
        if agentScoreV[imin] < agentcpScoreV[imin]:
            #print('WPF improved')
            return agentcp
        return agent


    def geneticEvo(self, xinit, yinit, step, length, mMT, rbT):
        # Generate initial population
        pop_size = 20*length
        pop = []
        popscore = []
        pScoreV = []
        for i in range(pop_size):
            pop.append(self.x_rw(xinit, yinit, step, length))
            popscore.append(self.RhoV(pop[i], mMT, rbT))
            pScoreV.append(self.pmin(popscore[i], mMT))
        
        converged = False
        maxiter = 100
        j = 0
        elite = np.argmax(pScoreV)
        stall = 0
        previousBest = -10

        while j < maxiter and pScoreV[elite] < 0 and stall < 3:
            #print('Improving Population')
            # Evaluate Population
            for i in range(pop_size):
                popscore[i] = self.RhoV(pop[i], mMT, rbT)
                pScoreV[i] = self.pmin(popscore[i], mMT)

            elite = np.argmax(pScoreV)

            print('n: ', j, ' Best: ', elite, ' ', pScoreV[elite], ' avg: ', np.mean(pScoreV), ' std: ', np.std(pScoreV))
            if pScoreV[elite] == previousBest:
                stall += 1
            else:
                stall = 0
            previousBest = pScoreV[elite]

            # Parents
            pIndices = np.argsort(pScoreV)

            # Generate Children
            for i in range(0, pop_size//3, 2):
                randint = np.random.randint(0, 6)
                randadd = np.random.randint(1,6)
                p1 = pIndices[-randint]
                p2 = pIndices[-(randint + randadd)]
                k1 = pIndices[i]
                k2 = pIndices[i+1]
                children = self.reproduce(pop, popscore, p1, p2, length)
                # Mutate Children
                pop[k1] = self.mutateAgent(children[0], popscore, length)
                pop[k2] = self.mutateAgent(children[1], popscore, length)

            for i in range(pop_size//3, pop_size-1):
                # Train Surivors
                k = pIndices[i]
                pop[k] = self.WPFT(pop[k], popscore[k], 50, mMT, rbT)

            j = j + 1
        pIndices = np.argsort(pScoreV)
        return pop[pIndices[-1]]

    def saveRes(self, sln, mMT, rbT, PATH = './results/'):
        # Save the results of a test to a file
        # Should modify the prints to match your scenario of interest

        # Make sure that filename is unique
        datestamp = str(datetime.date.today())
        path = PATH + datestamp + '/'
        # Create target directory & all intermediate directories if don't exists
        if not os.path.exists(path):
            os.makedirs(path)
            print("Directory " , path ,  " Created ")
        else:    
            print("Directory " , path ,  " already exists")   
        fid = open(path + 'output', 'a')
        fid.write(str(datetime.datetime.now().time()) + '\n')
        fid.write('Final Robustness: ' + mMT + ' ' + str(self.robustness(sln, mMT, rbT))+ '\n')
        fid.write('Final Robustness: ' + 'el' + ' ' + str(self.robustness(sln, 'el', rbT))+ '\n')
        fid.write('Final Robustness: ' + 'n' + ' ' + str(self.robustness(sln, 'n', rbT))+ '\n')
        fid.write('Solution')
        fid.write(str(self.plant(sln)) + '\n')
        fid.write('Robustness of predicates 1 (obstacle), 2 (eventually), 3 (speed)\n')
        fid.write(str(rob1.RhoV(sln, mMT, rbT)) + '\n')
        fid.write(str(r2.RhoV(sln, mMT, rbT))+ '\n')
        fid.write(str(r3.RhoV(sln, mMT, rbT))+ '\n')
        fid.write('Combined Robustness Vector\n')
        fid.write(str(self.RhoV(sln, mMT, rbT)) + '\n\n')
        fid.close()
        timestamp = str(datetime.datetime.now().time()).replace(':', '_')
        self.plotsln3D(sln, mMT, rbT, path + 'outputGraphs' + timestamp)

    def rectPtch(x1, x2, y1, y2, color='red'): 
        # Defines a patch object for easy plotting
        return plt.Rectangle((x1,y1), x2-x1, y2-y1, color=color, alpha=0.5)

if __name__ == '__main__':
    import scipy.optimize as sciOpt
    import datetime
    import os
    # Available Times
    t1 = 0
    t2 = 19 
    length = 20 
    step = 0.22

    test = 'n' #'simp' for simple, 'n' for number of time steps test. 'p' for disjuction predicate test
    op = 'op' # 'op' for optimize, 'dbg' for debug, 'dbghc' for debug hardcore
    optp = 'gen' # ef, agm, lse, gen

    # P-test Predicates
    pobs1 = ~STLpredicate.rect(t1,t2, 'a', 0.5, 1, 0.5, 1)
    pobs2 = ~STLpredicate.rect(t1,t2, 'a', -0.25, 0.25, 0.5, 1)
    pobs3 = ~STLpredicate.rect(t1,t2, 'a', 0.5, 1, -1, -0.5)

    obs1p = STLpredicate.rectPtch(0.5, 1, 0.5, 1)
    obs2p = STLpredicate.rectPtch(-0.25, 0.25, 0.5, 1)
    obs3p = STLpredicate.rectPtch(0.5, 1, -1, -0.5)

    pgas1 = STLpredicate.rect(t1, 10, 'e', 1.2, 1.5, -0.5, 0.25)
    pgas2 = STLpredicate.rect(t1, 10, 'e', 0, 0.3, 1.25, 1.5)
    pgas3 = STLpredicate.rect(t1, 10, 'e', -0.25, 0.25, -1.5, -1.75)

    gas1p = STLpredicate.rectPtch(1.2, 1.5, -0.5, 0.25, 'blue') 
    gas2p = STLpredicate.rectPtch(0, 0.3, 1.25, 1.5, 'blue') 
    gas3p = STLpredicate.rectPtch(-0.25, 0.25, -1.5, -1.75, 'blue') 

    pgoal1 = STLpredicate.rect(t1,t2, 'e', 1.5, 2, 1.5, 2)

    goal1p = STLpredicate.rectPtch(1.5, 2, 1.5, 2, 'green')

    # Complete P-Test predicates
    p = pobs1*pobs2*pobs3*pgoal1*(pgas1 + pgas2 + pgas3)

    # N-test Predicates
    rob1 = ~STLpredicate.rect(t1,t2, 'a', 0.5, 1, 0.5, 1)
    rob2 = ~STLpredicate.rect(t1,t2, 'a', 1.25, 2, 0.5, 1)
    rob3 = ~STLpredicate.rect(t1,t2, 'a', 0.5, 1, 1.25, 2)

    obs1n = STLpredicate.rectPtch(0.5, 1, 0.5, 1)
    obs2n = STLpredicate.rectPtch(1.25, 2, 0.5, 1)
    obs3n = STLpredicate.rectPtch(0.5, 1, 1.25, 2)
        
    rg1 = STLpredicate.rect(t1,t2, 'e', 1.5, 2, 1.5, 2)
    goal1n = STLpredicate.rectPtch(1.5, 2, 1.5, 2, 'green')

    # Complete N-test Scenario
    n = rob1*rob2*rob3*rg1

    # Simple Test Predicates
    spobs = ~STLpredicate.rect(t1,t2, 'a', 0.5, 1, 0.5, 1)
    spobsp = STLpredicate.rectPtch(0.5, 1, 0.5, 1)

    spgoal = STLpredicate.rect(t1,t2, 'e', 1.5, 2, 1.5, 2)
    spgoalp = STLpredicate.rectPtch(1.5, 2, 1.5, 2, 'green')
    
    sp = spobs*spgoal

    # Dynamics: Currently for TurtleBot Burger
    r3 = STLpredicate(t1,t2, 'a', np.array([0,0,1,0]), 0.22)
    r4 = STLpredicate(t1,t2, 'a', np.array([0,0,0,1]), 2.84)
    r5 = STLpredicate(t1,t2, 'a', np.array([0,0,-1,0]), 0.1) # to prevent reversing
    r6 = STLpredicate(t1,t2, 'a', np.array([0,0,0,-1]), 2.84)
    d = r3*r4*r5*r6

    if test == 'p':
        q = p*d
        print('Predicate Stress Test')
    elif test == 'n':
        q = n*d
        print('Time Step Stress Test')
    elif test == 'simp':
        q = sp*d
        print('Simple Test')
    else:
        print('Invalid Test')

    # Now time to run optimization
    if op == 'op':
        if optp == 'gen':
            rbT = 'pw'
            mMT = 'ef'
            print('Beginning Genetic Optimization. mMT: ', mMT, ' rbT: ', rbT)
            start = time.time()
            sln1 = q.geneticEvo(0,0,step,length, mMT, rbT)
            print('Beginning Vanilla Optimization')
            sln = q.VanillaMin(0,0,sln1)
            stop = time.time()
        
        elif optp == 'ef':
            rbT = 'n'
            mMT = 'ef'
            print('Beginning SLSQP Optimization. mMT: ', mMT, ' rbT: ', rbT)
            start = time.time()
            sln1 = q.x_rw(0,0,step,length)
            sln = q.VanillaMin(0,0,sln1, mMT=mMT, rbT=rbT, mthd='SLSQP')
            stop = time.time()

        # AGM optimization: again this doesn't quite work
        #elif optp == 'agm':
        #    rbT = 'n'
        #    mMT = 'ag'
        #    print('Beginning AGM-SLSQP Optimization. mMT: ', mMT, ' rbT: ', rbT)
        #    start = time.time()
        #    sln1 = q.x_rw(0,0,step,length)
        #    sln = q.VanillaMin(0,0,sln1, mMT=mMT, rbT=rbT, mthd='SLSQP')
        #    stop = time.time()
        #    print(q.robustness(sln, 'n', 'ag'))

        elif optp == 'lse':
            rbT = 'n'
            mMT = 'el'
            print('Beginning LSE-SLSQP Optimization. mMT: ', mMT, ' rbT: ', rbT)
            start = time.time()
            sln1 = q.x_rw(0,0,step,length)
            sln = q.VanillaMin(0,0,sln1, mMT=mMT, rbT=rbT, mthd='SLSQP')
            stop = time.time()

        else:
            print('Invalid Optimization type: ', optp)

        # Write Solution to file
        path = './sln.txt'
        fid = open(path, 'w')
        sln.tofile(fid)
        fid.close()

    elif op == 'dbg':
        # Debug mode pulls the previous trajectory from file. Its a huge timesaver.
        start = 0
        stop = 0
        sln = np.fromfile('./sln.txt')
        sln = sln.reshape((2,-1))
    
    elif op =='dbghc':
        # Hard Core debug mode uses a simple trajectory so you can probe specific elements
        start = 0
        stop = 0
        spd = 0.22*np.ones((1,length))
        omega = 0.01*np.ones((1,length))
        sln = np.vstack((spd,omega))
        print(sln)
    
    # Print Results on Screen
    if test == 'p':
        rbT = 'pw'
        print('Optimization Time: ', stop - start)
        print('Final Robustness  n-n: ', q.robustness(sln, 'n', 'n'))
        print("Solution")
        slnCmplt = q.plant(sln)
        print(slnCmplt)
        print("Robustness of obstacles")
        print((pobs1.RhoV(sln, 'n', rbT)))
        print((pobs2.RhoV(sln, 'n', rbT)))
        prtnt((pobs3.RhoV(sln, 'n', rbT)))
        print('Robustness of gas')
        print((pgas1.RhoV(sln, 'n', rbT)))
        print((pgas2.RhoV(sln, 'n', rbT)))
        print((pgas3.RhoV(sln, 'n', rbT)))
        print('Robustness of Eventually')
        print((pgoal1.RhoV(sln, 'n', rbT)))
        print('Combined Robustness Vector Normal')
        print(q.RhoV(sln, 'n', rbT))
        
        fix, ax = plt.subplots(1)
        ax.set_xlim((-1,2))
        ax.set_ylim((-2,2))

        ax.add_patch(obs1p)
        ax.add_patch(obs2p)
        ax.add_patch(obs3p)

        ax.add_patch(gas1p)
        ax.add_patch(gas2p)
        ax.add_patch(gas3p)

        ax.add_patch(goal1p)
        ax.plot(slnCmplt[0,:],slnCmplt[1,:], linestyle='-', marker="o")
        plt.title('Predicate Test')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    elif test == 'n':
        rbT = 'pw'
        print('Optimization Time: ', stop - start)
        print('Final Robustness  n-n: ', q.robustness(sln, 'n', 'n'))
        print("Solution")
        slnCmplt = q.plant(sln)
        print(slnCmplt)

        print("Robustness of obstacles")
        print((pobs1.RhoV(sln, 'n', rbT)))
        print((pobs2.RhoV(sln, 'n', rbT)))
        print((pobs3.RhoV(sln, 'n', rbT)))

        print('Robustness of Eventually')
        print((pgoal1.RhoV(sln, 'n', rbT)))

        print('Combined Robustness Vector Normal')
        print(q.RhoV(sln, 'n', rbT))

        fix, ax = plt.subplots(1)
        ax.set_xlim((-0.5,2))
        ax.set_ylim((-0.1,2.5))

        ax.add_patch(obs1n)
        ax.add_patch(obs2n)
        ax.add_patch(obs3n)

        ax.add_patch(goal1n)
        ax.plot(slnCmplt[0,:],slnCmplt[1,:], linestyle='-', marker="o")

        plt.title('Time Step Test, N=' + str(length))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    elif test == 'simp':
        rbT= 'pw'
        print('Optimization Time: ', stop - start)
        print('Final Robustness  n-n: ', q.robustness(sln, 'n', 'n'))
        print("Solution")
        slnCmplt = q.plant(sln)
        print(slnCmplt)

        print("Robustness of obstacle")
        print((spobs.RhoV(sln, 'n', rbT)))

        print('Robustness of Eventually')
        print((spgoal.RhoV(sln, 'n', rbT)))

        fix, ax = plt.subplots(1)
        ax.set_xlim((-0.5,2))
        ax.set_ylim((-0.1,2.5))

        ax.add_patch(spobsp)
        ax.add_patch(spgoalp)

        ax.plot(slnCmplt[0,:],slnCmplt[1,:], linestyle='-', marker="o")

        plt.title('Simple-Test')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

