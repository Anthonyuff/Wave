import matplotlib.pyplot as plt
import numpy as np
import segyio
from scipy.ndimage import zoom


class Model:
    
    def __init__(self, c: "Config"):
        
        self.c= c
        self.model= np.zeros((self.c.nz,self.c.nx))
        
        self.dh = 0
        self.dt = 0
 
        self.nz_util = 351
        self.nx_util = 1701

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

        self.marmo=np.array((self.c.nz,self.c.nx))
        
        
    def geo(self) -> None :
        
        self.offset = int(20 / self.dh) 
        
        self.rx = list(range(20, self.c.nx-self.c.nabc, self.offset))
        self.rz=[20]*len(self.rx)
        
        self.rzf = [2]*len(self.rx)
        self.rec = np.zeros((self.c.nt, len(self.rz)))

        self.rz1 = list(range(20, self.c.nz-self.c.nabc, self.offset))
        self.sx= self.c.nx//2
    
    def geoaqui(self) -> None :
        
        self.sx= np.linspace(1000,16000,36)
        self.sz = 280*np.ones(len(self.sx),dtype= 'float32')

        self.rx= np.linspace(50,16950,170)
        self.rz = 450*np.ones(len(self.rx),dtype= 'float32')
        
        fontes = np.column_stack((self.sx, self.sz))
        np.savetxt(
        "./config/fontes.txt",
        fontes,
        fmt="%.2f",
        header="sx sz",
        comments=""
        )

        # Salvar receptores em txt
        receptores = np.column_stack((self.rx, self.rz))
        np.savetxt(
            "./config/receptores.txt",
            receptores,
            fmt="%.2f",
            header="rx rz",
            comments=""
        )
          
    def importe(self)-> None: 
        # self.marmo=np.fromfile('./data/Marmousi2.bin',dtype=np.float32)

        # self.marmo= self.marmo.reshape((self.c.nz,self.c.nx),order='F')
        # print(self.marmo.shape)

        arquivo = "./data/MODEL_P-WAVE_VELOCITY_1.25m.segy"

        with segyio.open(arquivo, "r", ignore_geometry=True) as f:
            vel = segyio.tools.collect(f.trace[:])

        print("Formato original:", vel.shape)

        vel = vel.T

        print("Formato como matriz:", vel.shape)

        self.nz_util = 351
        self.nx_util = 1701

        # fator de redimensionamento
        fator_z = self.nz_util/ vel.shape[0]
        fator_x = self.nx_util / vel.shape[1]

        # redimensiona para 396 x 1701
        self.marmo_util = zoom(vel, (fator_z, fator_x), order=1)

        # força o tamanho, só por segurança
        self.marmo_util = self.marmo_util[:self.nz_util, :self.nx_util].astype("float32")

        self.nabc = self.c.nabc


        # define dh como 10 m
        self.marmo = np.pad(
                            self.marmo_util,
                            pad_width=((self.nabc, self.nabc), (self.nabc, self.nabc)),
                            mode="edge"
                            ).astype("float32")

    # agora muda para o tamanho total da propagação
        self.c.nz = self.marmo.shape[0]
        self.c.nx = self.marmo.shape[1]
        print(self.dh)
        print("Modelo útil:", self.marmo_util.shape)
        print("Modelo com borda:", self.marmo.shape)
        print("nz total:", self.c.nz)
        print("nx total:", self.c.nx)

       
    def  disp(self) -> None :
        
        cmax = np.max(self.c.v.values)
        fmax = np.max(self.c.s.f0)
        self.dh = 10
        #self.dh = cmax / (self.alpha * fmax)
        
        self.dt=self.dh/(self.beta*cmax)
        print(self.dt)
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
        
        plt.imshow(self.marmo_util, aspect="auto", extent=[0, self.nx_util * self.dh, self.nz_util  * self.dh, 0])
        
        plt.scatter(self.sx , self.sz ,c= "green", marker="*", zorder=10,s=120,label="Source")
        #plt.scatter(np.array(self.sx) * self.dh +40 , np.array(self.szf)*self.dh ,c= "green", marker="*", zorder=10,s=120)
        plt.scatter(self.rx  , self.rz ,c= "red", s=10, label="Receptors")
        
        plt.colorbar(label="Velocity (m/s)")
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.legend()
        plt.title("Velocity Model")
        plt.show()
        