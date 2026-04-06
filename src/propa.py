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
    def __init__(self, c: "Config", m: "model"):
        
        self.c= c
        self.m = m
        self.cerjan = np.ones(self.c.nz,self.c.nx)  
        self.bord = np.zeros(self.c.nabc)
        self.sb = 1.5 * self.c.nabc
        self.P = np.zeros((self.c.nz,self.c.nt))  
    
    def ccerjan(self):
        
        for i in range(self.c.nabc):
            dist = self.c.nabc - i
            fb = dist / (1.4142 * self.sb)
            self.bord[i] = np.exp(-(fb * fb) )

        for ix in range(self.c.nx):    
            
            self.cerjan[:self.c.nabc,ix] *= self.bord
            self.cerjan[-self.c.nabc:,ix] *= self.bord[::-1]

        for iz in range(self.c.nz):    
            
            self.cerjan[iz,:self.c.nabc,] *= self.bord
            self.cerjan[iz,-self.c.nabc:] *= self.bord[::-1]
    
    def eq2D(self):

        
        d2u_dx2 = np.zeros((self.c.nz, self.c.nx))
        d2u_dz2 = np.zeros((self.c.nz, self.c.nx))

        ricker1= wavelet(self.c.s.f0,self.m.time)
        
        #criar matriz de snapshots
        snap=np.zeros((self.c.nz,self.c.nx,500))
        
        dh2 = self.m.dh * self.m.dh
        
        cte = (self.m.model * self.m.dt)**2
        s=0
        
        

        
        
        simo=np.zeros((self.c.nt,len(self.m.rx)))
        
        for t in range(1, self.c.nt-1):
            #dlay= 150 #delay
            # fonte
            self.P[20, self.m.sx, t] += ricker1[t] / dh2
            # if t>=dlay:
            #     P[20, sx+40, t] += fonte[t] / dh2
            
            


            laplacian = laplacian2d(
                self.P[:, :, t], d2u_dx2, d2u_dz2, self.c.nz, self.c.nx, dh2
            )


            self.P[:, :, t+1] = cte * laplacian + 2*self.P[:, :, t] - self.P[:, :, t-1]

            # salvar snapshots a um certo passo de tempo
            self.P[:, :, t] *= self.cerjan
            self.P[:, :, t+1] *= self.cerjan
            
            if t%4==0 and s<500:
                
                snap[:,:,s] = self.P[:,:,t]
                s += 1

            for j in range(len(rx)):
                    
                simo[t,j] = self.P[self.m.rz[j],self.m.rx[j], t]        

    def plot(self):
        
        #plot cerjan
        plt.imshow(self.cerjan, ascpect = "auto")
        plt.colorbar()
        plt.title('Cerjan')
        plt.show()

        plt.imshow(self.P[:,:,200],cmap='gray',aspect='auto', extent=[0, 9, nt*dt, 0], vmax=vmax, vmin=vmin)

@njit(parallel=True)
def laplace1D(P,t,dz,nz):
    
    lap = np.zeros(nz)
    for n in range(2, nz-2):
        
        lap[n] = (-P[n+2, t] + 16*P[n+1, t] - 30*P[n, t] + 16*P[n-1, t] - P[n-2, t])  / (12 * dz**2)
    
    return lap


@njit(parallel=True)   
def laplacian2d(upre, d2u_dx2, d2u_dz2, nzz, nxx, dh2) :
  inv_dh2 = 1.0 / (5040.0 * dh2)

  for i in prange(4, nzz - 4):
    for j in range(4, nxx - 4):
      d2u_dx2[i, j] = (
          -9   * upre[i-4, j] + 128   * upre[i-3, j] - 1008 * upre[i-2, j] +
          8064 * upre[i-1, j] - 14350 * upre[i,   j] + 8064 * upre[i+1, j] -
          1008 * upre[i+2, j] + 128   * upre[i+3, j] - 9    * upre[i+4, j]
      ) * inv_dh2

      d2u_dz2[i, j] = (
          -9   * upre[i, j-4] + 128   * upre[i, j-3] - 1008 * upre[i, j-2] +
          8064 * upre[i, j-1] - 14350 * upre[i, j]   + 8064 * upre[i, j+1] -
          1008 * upre[i, j+2] + 128   * upre[i, j+3] - 9    * upre[i, j+4]
      ) * inv_dh2

  return d2u_dx2 + d2u_dz2

def wavelet(freq,t):
  f_corte = freq

  fc = f_corte / (3 * np.sqrt(np.pi))

  td = t - (0.5 * np.sqrt(np.pi) / fc)

  arg = np.pi * (np.pi * fc * td)**2

  return (1 - 2*arg) * np.exp(-arg)