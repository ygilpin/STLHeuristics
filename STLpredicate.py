#!/usr/bin/env python3

import numpy as np

class STLpredicate:
    def __init__(self, t1, t2, ae, A, b, cdn = 0 , other = 0):
        self.t1 = t1 # When predicate is active
        self.t2 = t2
        self.A  = A
        self.b  = b
        self.ae = ae # always or eventually
        self.cdn = cdn # conjuction, disjunction or negation
        self.other = other # other STL predicate for conjuction/disjunction

    def myRho(self, x, t):
        #print("myRho called")
        # Calculate the robustness of myself
        if self.t1 <= t and t <= self.t2:
            if self.ae == 'a':
                self.rhoVal = self.b - self.A @ x[:,t]
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
                    self.rhoVal = abs(deltat)/(abs(deltax)+1)
                else:
                    self.rhoVal = -abs(deltax)/(abs(deltat)+1)
            if self.cdn == 'n':
                self.rhoVal = -self.rhoVal

        else: 
            self.rhoVal = "Nan"
        return self.rhoVal

    def Rho(self, x,t):
        #print("Robustness function called")
        # Force the calcluation of my robustness
        self.myRho(x, t)
        # Now apply conjuction and disjunction
        if self.other:
            print("getting other robustness")
            rho_o = self.other.Rho(x,t);
            if self.cdn == 'c':
                if self.rhoVal != "Nan" and rho_o != "Nan":
                    return min(self.rhoVal, rho_o)
                elif self.rhoVal == "Nan" and rho_o != "Nan":
                    return rho_o
                elif self.rhoVal != "Nan" and rho_o == "Nan":
                    return self.rhoVal
                else:
                    return "NaN"
            elif self.cdn == 'd':
                if self.rhoVal != "Nan" and rho_o != "Nan":
                    print("Maximizing robustness")
                    return max(self.rhoVal, rho_o)
                elif self.rhoVal == "Nan" and rho_o != "Nan":
                    return rho_o
                elif self.rhoVal != "Nan" and rho_o == "Nan":
                    return self.rhoVal
                else:
                    return "NaN"
        else:
            return self.rhoVal

    def __invert__(self):
        return STLpredicate(self.t1, self.t2, self.ae, self.A, self.b, 'n', 0)

    def __add__(self, other):
        return STLpredicate(other.t1, other.t2, other.ae, other.A, other.b, 'd', self)

    def __mul__(self, other):
        return STLpredicate(other.t1, other.t2, other.ae, other.A, other.b, 'c', self)

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
A1 = np.array([1, 2])
b1 = 5
A2 = np.array([3, -5])
b2 = 3
t = np.linspace(t1, t2, t2 + 1)
x = np.vstack((t,2*t))

x1 = 1
x2 = 3
y1 = 1
y2 = 3
ae = 'a'
p1 = STLpredicate(t1, t2, ae, np.array([0, -1]), -y2)
        # Bottom line
p2 = STLpredicate(t1,t2,ae,np.array([0, -1]), y1)
        # Left line
p3 = STLpredicate(t1,t2,ae,np.array([1, 0]), x1)
        # Right line
p4 = STLpredicate(t1,t2,ae,np.array([-1, 0]), -x2)
print("Testing the robustness of rectangle")
for time in range(t1,t2+1):
    print(x[:,time])
    print('p1', p1.Rho(x,time))
    print('p2', p2.Rho(x,time))
    print('p3', p3.Rho(x,time))
    print('p4', p4.Rho(x,time))
    print(x[:,time], p.Rho(x,time))
    print(x[:,time], p5.Rho(x,time))"""
