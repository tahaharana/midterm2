import numpy as np
import scipy.linalg

#----------------------------------- MAXIMAL RELATIVE ERROR FOR AX = B -----------------------------------------
A = np.array([[-1.0, 16.0, 0.0], [0.0, 4.0, 10.0], [1.0, -4.0, 40.0]])
b = np.array([[17.0], [14.0], [35.0]])

x = np.array([[-0.99900385], [0.99991435], [1.0000192]])
x_exact = np.linalg.solve(A, b)
print(x_exact)

result = A@x - b

cond_num = np.linalg.cond(A)
r_norm = scipy.linalg.norm(result)
b_norm = scipy.linalg.norm(b)

rel_error = cond_num * (r_norm/b_norm)

print(rel_error)
print('\n')

#-------------------------------------- Ax = b SOLVING LINEAR SYSTEMS ------------------------------------------
A = ([1.3, 2.0, -1.3], [2.3, 0.1, -1.1], [0.2, 2.1, -0.2])
b = ([0.4], [0.1], [1.2])
Pt,L,U = scipy.linalg.lu(A)
P = Pt.T
Pb = np.matmul(P,b)
y = scipy.linalg.solve_triangular(L,Pb,lower=True)
x = scipy.linalg.solve_triangular(U,y,lower=False)
print(x)
print('\n')


# ----------------------------------- PA = LU DECOMPOSITION -------------------------------------------
# PA = LU decomposition
# A = np.array([[2,1,1,0], [4,3,3,1], [8,7,9,5], [6,7,9,8]])
A = np.array([[2.4, -3.3, -1.0], [1.5, -1.2, 0.3], [2.2, 2.9, 1.3]])

Pt,L,U = scipy.linalg.lu(A)
P = Pt.T

print(L)
print('\n')
print(U)
print('\n')
print(P)
print('\n')

perm = np.array([1,2,3])

print(np.matmul(perm, P))



# -------------------------------------- X3=14 ITERATION ------------------------------------------------
def phi(x):
    # Add code
	return  3*x-1


def iteration(f, initial, endpoint):
    for i in range(endpoint):
        y = f(initial)
        temp = y
        initial = temp
        

    return y

x0 = 1
iter = 3
print(iteration(phi, x0, iter))