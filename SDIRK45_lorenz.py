from pylab import *
import numpy as np
import time
from numpy.linalg import inv
from numpy.linalg import norm

################ THE LORENZ EQUATIONS ################
sigma = 10.0; rho = 28.0; beta = 8.0/3.0

def Lorenz(y):
    return np.array([sigma*(y[1]-y[0]),y[0]*(rho-y[2])-y[1],y[0]*y[1]-beta*y[2]])
    
def Lorenz_Jac(y):
    return np.array([[-sigma,sigma,0],[rho-y[2],-1,-y[0]],[y[1],y[0],-beta]])
######################################################

####### THE BUTCHER TABLEAU OF SDIRK45 ###############
C = np.array([1/4, 3/4, 11/20, 1/2, 1])
A = np.array([
    [1/4,      0,         0,      0,      0],
    [1/2,      1/4,       0,      0,      0],
    [17/50,    -1/25,     1/4,    0,      0],
    [371/1360, -137/2720, 15/544, 1/4,    0],
    [25/24,    -49/48,    125/16, -85/12, 1/4]
])
B = np.array([25/24, -49/48, 125/16, -85/12, 1/4])
######################################################

h=0.01           #time-step size

y0 = np.array([1.5,2.5,15])  #initial conditions
ODE = Lorenz
Jac = Lorenz_Jac

Neq = 3          #the number of equations
Nstage = 5       #the number of stages in the scheme
Nexp = 3         #the number of Lyapunov exponents to compute

Tol = 1.0e-12    #tolerance

I = np.identity(Neq)

k = np.zeros([Nstage,Neq])         #Runge-Kutta slopes
kw = np.zeros([Nexp,Nstage,Neq])   #Runge-Kutta slopes for the tangent equation

tmax_approx = 20000                #total time
Nstep = int(tmax_approx/h)         #total number of timesteps
t = np.arange(0,Nstep*h,h)
MaxIter = 80000                    #maximum number of iterations

t_trans = tmax_approx/2.0          #time to start computing the exponents

y_old = np.zeros_like(y0)          #solution at the previous time-step
np.copyto(y_old,y0)

w_old = np.identity(Neq)[0:Nexp]   #tangent vectors at the previous time-step
w = np.zeros_like(w_old)

e = np.zeros(Nexp)
tlam = []                          #time for Lyapunov exponents
lam = []                           #Lyapunov exponents

lam_output_time = 1.0              #output time of the exponents
lam_output_count = 0

CPU_time = []

for i in range(len(t)):
    Jac_old = Jac(y_old)
    
    stp = time.process_time()   #record initial CPU time
    
    for st in range(Nstage):
    
        k[st] = ODE(y_old + h*np.dot( A[st,:st], k[:st] ))  #initialize k
        for j in range(MaxIter):
            F = k[st] - ODE(y_old + h*np.dot( A[st,:st+1], k[:st+1] ) )
            J_F = I - h*A[st,st]*Jac(y_old + h*np.dot( A[st,:st+1], k[:st+1] ) )
            k_inc = np.dot(inv(J_F),F)
            if norm(k_inc) < Tol: break
            k[st] += -k_inc
            if j==MaxIter-1: print('Did not converge in maximum number of iterations'); exit()
            
    etp = time.process_time()   #record end CPU time
    CPU_time.append(etp-stp)
            
    y_new = y_old + h*np.dot( B, k )
    np.copyto(y_old,y_new)
        
    if t[i]>t_trans:
        for st in range(Nstage):
            for m in range(len(kw)):
                kw[m][st] = np.dot( inv(I - h*A[st,st]*Jac_old), \
                    np.dot( Jac_old, w_old[m] + h*np.dot( A[st,:st], kw[m][:st] ) ) )
        for m in range(len(kw)):
            w[m] = w_old[m] + h*np.dot( B, kw[m] )
            
        wp = []
        for l in range(len(w)):
            wp_x = np.zeros(len(w[l]))
            np.copyto(wp_x,w[l])
            for m in range(l-1,-1,-1):
                wp_x += -np.dot(w[l],wp[m])*wp[m]
            wp.append(wp_x)
        wp = np.array(wp)
                
        e = e + log(norm(wp,axis=1))
        
        if int((t[i]-t_trans)/lam_output_time)>lam_output_count:
            tlam.append(t[i])
            lam.append(e/(t[i]-t_trans))
            lam_output_count += 1
        
        for m in range(len(w)):
            w[m] = wp[m]/norm(wp[m])
            
        np.copyto(w_old,w)
    
    
    
outfile = open('./lorenz_lambdas.dat',"w")
for i in range(len(tlam)):
    outfile.write(str(tlam[i]) + '   ')
    for m in range(Nexp):
        outfile.write(str(lam[i][m]) + '   ')
    outfile.write('\n')
outfile.close()

