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
                deltax = self.b - self.A @ x[:,t]
                deltat = self.t2 -t
                if sat:
                    rhoVal = abs(deltat)/(abs(deltax)+1)
                else:
                    rhoVal = -abs(deltax)/(abs(deltat)+1)
            if self.cdn == 'n':
                rhoVal = -rhoVal

        print("myRho called", rhoVal)
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
            return left.Rho(x,t)
        if self.right:
            return right.Rho(x,t)
        return self.myRho(x,t)

    def __invert__(self):
        return STLpredicate(self.t1, self.t2, self.ae, self.A, self.b, 'n', 0)

    def __add__(self, other):
        print("Disjunction between two predicates")
        return STLpredicate(cdn='d', left=self, right=other)

    def __mul__(self, other):
        print("Conjunction between two predicates")
        return STLpredicate(cdn='c', left=self, right=other)

    def arect(t1,t2,ae,x1,x2,y1,y2):
        # First the equations for the lines are needed
        # Top line
        p1 = STLpredicate(t1, t2, ae, np.array([0, -1]), -y2)
        # Bottom line
        p2 = STLpredicate(t1,t2,ae,np.array([0, -1]), y1)
        # Left line
        p3 = STLpredicate(t1,t2,ae,np.array([1, 0]), x1)
        # Right line
        p4 = STLpredicate(t1,t2,ae,np.array([-1, 0]), -x2)
        return p1 + p2 + p3 + p4

t1 = 0
t2 = 4  
time = 2
A1 = np.array([1, 2])
b1 = 5
A2 = np.array([3, -5])
b2 = 3
t = np.linspace(t1, t2, t2 + 1)
x = np.vstack((t,2*t))

p = STLpredicate.arect(t1, t2, 'a', 1, 3, 1, 3)
for time in range(t1,t2+1):
    print(x[:,time])
    print(x[:,time], p.Rho(x,time))
