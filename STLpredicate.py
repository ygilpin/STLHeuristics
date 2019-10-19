#!/usr/bin/env python3

import numpy as np

class STLpredicate:
    def __init__(self, t1 = 0, t2 = 0, ae = 0, A = 0, b = 0, cdn = 0 , left = 0, right = 0):
        self.t1 = t1 # When predicate is active
        self.t2 = t2
        self.A  = A # Equation of a line
        self.b  = b # constant from ^
        self.ae = ae # always or eventually
        self.cdn = cdn # conjuction, disjunction or negation
        self.left = left # other STL predicates for conjuction/disjunction
        self.right = right

    def myRho(self, x, t):
        # Calculate the robustness of myself
        # Should only be called when I have no left or right branch
        rhoVal = 'Nan'
        if self.t1 <= t and t <= self.t2:
            if self.ae == 'a':
                rhoVal = self.b - self.A @ x[:,t]
            else:
                # Check if there is satisfaction at any point
                sat = False
                for k in range(self.t1,self.t2+1):
                    if (self.b - self.A @ x[:,k]) > 0:
                        sat = True
                        break
                deltax = abs(self.b - self.A @ x[:,t])
                deltat = abs(1 + self.t2 -t)
                #print(deltax)
                #print(deltat)
                if sat:
                    rhoVal = (deltat)/((deltax)+0.15)
                else:
                    rhoVal = -(deltax)/((deltat)+0.15)
        #print("myRho called", rhoVal)
        return rhoVal

    def Rho(self, x,t):
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

    def robustness(self,x):
        p = []
        for t in range(self.t1, self.t2 + 1):
            p.append(self.Rho(x,t))
        return min(p)

    def robustnessflt(self,xflt):
        # Same robustness, but flattened version
        x = np.reshape(xflt, (2,-1)) 
        p = []
        for t in range(self.t1, self.t2 + 1):
            p.append(self.Rho(x,t))
        return min(p)

    def __invert__(self):
        return STLpredicate(t1=self.t1, t2=self.t2, cdn='n', left=self)

    def __add__(self, other):
        #print("Disjunction between two predicates")
        return STLpredicate(t1=min(self.t1, other.t2), t2=max(self.t2,other.t2), cdn='d', left=self, right=other)

    def __mul__(self, other):
        #print("Conjunction between two predicates")
        return STLpredicate(t1=min(self.t1, other.t2), t2=max(self.t2,other.t2), cdn='c', left=self, right=other)

    def rect(t1,t2,ae,x1,x2,y1,y2):
        # First the equations for the lines are needed
        # Top line
        p1 = STLpredicate(t1, t2, ae, np.array([0, 1]), y2)
        # Bottom line
        p2 = STLpredicate(t1,t2,ae,np.array([0, -1]), -y1)
        # Left line
        p3 = STLpredicate(t1,t2,ae,np.array([-1, 0]), -x1) 
        # Right line
        p4 = STLpredicate(t1,t2,ae,np.array([1, 0]), x2)
        return p1 * p2 * p3 * p4

    def cost(self, xflt):
        x = np.reshape(xflt, (2, -1))
        p = self.robustness(x)
        d = 0
        for t in range(t1,t2):
            deltax = x[:,t] - x[:,t+1] 
            d += np.linalg.norm(deltax)
        return 1*d - p*(self.t2 - self.t1 + 1)
    
    def bounds(self, x0, y0):
        # Returns a tuple for the bound constraints
        # The first point is fixed. 
        bnds = []
        for i in range(1,3):
            if i == 1:
                bnds.append((x0,x0))
            elif i == 2:
                bnds.append((y0, y0))
            for j in range(t1+1,t2+1):
                bnds.append((None,None))
        return bnds 

    def x_guess(self, x0, y0):
        # The intial point is x0, y0
        # Random trajectory after that
        x_tail = 9*np.random.rand(2, self.t2 - self.t1)
        x_start = np.array([[x0], [y0]])
        x = np.concatenate((x_start, x_tail), axis=1)
        return x


if __name__ == '__main__':
    from scipy.optimize import minimize
    # Available Times
    t1 = 0
    t2 = 14 

    # Some predicates based on these points
    r1 = ~STLpredicate.rect(t1,t2, 'a', 1, 3, 1, 3)
    r2 = STLpredicate.rect(t1,t2, 'e', 6, 9, 6, 9)
    p = r1*r2

    # Generate a guess trajectory 
    x1 = p.x_guess(0,0)
    guess = x1.flatten()
    print('Initial Guess: ')
    print(x1)
    print('Initial Guess Robustness: ', p.robustness(x1))
    print('Initial cost: ', p.cost(x1))
    # Now time to run optimization
    bnd = p.bounds(0, 0)
    sln = minimize(p.cost, guess, method='TNC', bounds=bnd, tol=1e-6, options =
            { 'disp':True
                })
    print(np.reshape(sln.x, (2,-1)))
    print('Final Robustness: ', p.robustnessflt(sln.x))
    print('Final cost: ', p.cost(sln.x))

