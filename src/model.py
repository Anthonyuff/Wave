import matplotlib.pyplot as plt
import numpy as np


class Model:
    
    def __init__(self, c: "Config"):
        
        self.c= c
        self.model= np.zeros((self.c.nz,self.c.nx))
        
        self.dh = 0
        self.dt = 0

        self.alpha= 3
        self.beta= 4
        self.sx = 0
        self.sz = 20
        self.szf = [2]
        self.offset = 0
        
        self.rx = [0]
        self.rz = [0]
        self.rzf = [0]

        self.rz1 = [0]
        
        
    def geo(self) -> None :
        
        self.offset = int(20 / self.dh) 
        
        self.rx = list(range(20, self.c.nx-self.c.nabc, self.offset))
        self.rz=[20]*len(self.rx)
        
        self.rzf = [2]*len(self.rx)
        self.rec = np.zeros((self.c.nt, len(self.rz)))

        self.rz1 = list(range(20, self.c.nz-self.c.nabc, self.offset))
        self.sx= self.c.nx//2

          


    def  disp(self) -> None :
        
        cmax = np.max(self.c.v.values)
        fmax = np.max(self.c.s.f0)

        self.dh = cmax / (self.alpha * fmax)

        self.dt=self.dh/(self.beta*cmax)

        self.time = np.arange(0,self.c.nt*self.dt,self.dt)
        self.depth = np.arange(0,self.c.nz*self.dh,self.dh)
    
    def create(self) -> None:
        
        if len(self.c.v.interfaces) == 0:
                self.model[:, :] = self.c.v.values[0]
        else:
                self.model[:self.c.v.interfaces[0], :] = self.c.v.values[0]

                for i in range(1, len(self.c.v.interfaces)):
                    z_ini = self.c.v.interfaces[i - 1]
                    z_fim = self.c.v.interfaces[i]
                    self.model[z_ini:z_fim, :] = self.c.v.values[i]

                
                self.model[self.c.v.interfaces[-1]:, :] = self.c.v.values[-1]
    def plotmodel(self):
        
        plt.imshow(self.model, aspect="auto", extent=[0, self.c.nx * self.dh, self.c.nz * self.dh, 0])
        
        plt.scatter(np.array(self.sx) * self.dh , np.array(self.szf)*self.dh ,c= "green", marker="*", zorder=10,s=120,label="Source")
        plt.scatter(np.array(self.rx) * self.dh , np.array(self.rzf)*self.dh ,c= "red", s=10, label="Receptors")
        
        plt.colorbar(label="Velocity (m/s)")
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.legend()
        plt.title("Velocity Model")
        plt.show()
        