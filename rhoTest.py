#!/usr/bin/env python3

import STLpredicate as stl
import numpy as np
from scipy.optimize import minimize 

t1 = 0
t2 = 7
x1 = np.array([[0,1,2,3,4,5,6,7,7.5], [0,1,2,3,4,5,6,7,7.5]])
print(stl)
stl.myprint()
p2 = stl(t1,t2, 'a', np.array([0,1]), 2)
r1 = ~stl.rect(t1,t2, 'a', 1, 3, 1, 3)
r2 = stl.rect(t1,t2,'e', 6, 9, 6, 9)
p = r1*r2
