import time
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange

def measure_runtime(func):
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"Runtime: {round(end - start, 4)} seconds")
    return result
  return wrapper

class Wave1D:
    
    def __init__(self, c: "Config", m: "model"):
        self.c= c
        self.m = m
        self.cerjan = np.ones(self.c.nz)  
        self.bord = np.zeros(self.c.nabc)
        self.sb = 1.5 * self.c.nabc
        self.P = np.zeros((self.c.nz,self.c.nt))  
        
    
    def ccerjan(self) ->  None:
        for i in range(self.c.nabc):
            dist = self.c.nabc - i 
            fb = dist / (1.4142 * self.sb)
            self.bord[i] = np.exp(-(fb * fb)*0.55)
        self.cerjan[:self.c.nabc] = self.bord
        self.cerjan[-self.c.nabc:] = self.bord[::-1]
    
    @measure_runtime
    def eq1d(self) ->  None:
        
        sz= self.c.nz//2
        ricker1= wavelet(self.c.s.f0,self.m.time)
        
        for i in range(1,self.c.nt-1):

            self.P[sz,i] += ricker1[i]

            laplacian = laplace1D(self.P,i,self.m.dh,self.c.nz)

            self.P[:,i+1] = (self.m.model[:,0]*self.m.dt)**2 * laplacian + 2*self.P[:,i] - self.P[:,i-1]  
            
            self.P[:, i] *= self.cerjan

            self.P[:, i+1] *= self.cerjan

            for j in range(len(self.m.rz1)):
                    
                    self.m.rec[i,j] = self.P[self.m.rz1[j], i]
    
    def plot(self) ->  None:
        plt.plot(self.cerjan)

        plt.show()  

        plt.imshow(self.P.T, aspect='auto', cmap='gray', extent=[0, self.c.nt*self.m.dt, self.c.nz*self.m.dh, 0])

        plt.xlabel('time (s)')

        plt.ylabel('death (m)')

        plt.show() 

    def animation(self) ->  None:
        from matplotlib.animation import FuncAnimation

        fig,ax = plt.subplots()
        linha, = ax.plot(self.m.depth, self.P[:,0])
        ax.set_xlim(self.m.depth.min(), self.m.depth.max())
        ax.set_ylim(-self.P.max(),self.P.max() )

        def atualizar(frame):
            linha.set_ydata(self.P[:,frame])
            ax.set_title(f"time = {frame*self.m.dt:.3f} s")
            return linha,

        ani = FuncAnimation(fig, atualizar, frames=self.c.nt, interval=10)
        #ani.save('onda1d.gif',writer='pilow',fps=30)
        plt.show()                 


         
class Wave2D:
    pass


@njit(parallel=True)
def laplace1D(P,t,dz,nz) -> None:
    
    lap = np.zeros(nz)
    for n in range(2, nz-2):
        
        lap[n] = (-P[n+2, t] + 16*P[n+1, t] - 30*P[n, t] + 16*P[n-1, t] - P[n-2, t])  / (12 * dz**2)
    
    return lap 

def wavelet(freq,t):
  f_corte = freq

  fc = f_corte / (3 * np.sqrt(np.pi))

  td = t - (0.5 * np.sqrt(np.pi) / fc)

  arg = np.pi * (np.pi * fc * td)**2

  return (1 - 2*arg) * np.exp(-arg)