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
                    rhoVal = (deltat)/((deltax)+1)
                else:
                    rhoVal = -(deltax)/((deltat)+1)
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

    def __invert__(self):
        return STLpredicate(cdn='n', left=self)

    def __add__(self, other):
        #print("Disjunction between two predicates")
        return STLpredicate(cdn='d', left=self, right=other)

    def __mul__(self, other):
        #print("Conjunction between two predicates")
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

# Available Times
t1 = 0
t2 = 0  

# Available lines
A1 = np.array([1, 0])
b1 = 3
A2 = np.array([0, 1])
b2 = 3
A3 = np.array([0, -1])
b3 = -3

# Available Test points
x1 = np.array([[1], [1]])
x2 = np.array([[2], [1]])
x3 = np.array([[4], [1]])

# Some predicates based on these points
p1 = STLpredicate(t1, t2, 'e', A1, b1)
p2 = STLpredicate(t1, t2, 'e', A2, b2)
p3 = STLpredicate(t1, t2, 'e', A3, b3)
p4 = p1 + p2 + p3
p5 = p1*p2*p3
p6 = p1*(p2 + p3)
p7 = p1*(p1 + p3)
p8 = ~p7 * p3

# Lets check the robustness
print("p1 ", p1.Rho(x1,0))
print("p2 ", p2.Rho(x1,0))
print("p3 ", p3.Rho(x1,0))
print("p4 ", p4.Rho(x1,0))
print("p5 ", p5.Rho(x1,0))
print("p6 ", p6.Rho(x1,0))
print("p7 ", p7.Rho(x1,0))
print("p8 ", p8.Rho(x1, 0))
