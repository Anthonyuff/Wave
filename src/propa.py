import time
import matplotlib.pyplot as plt
import numpy as np
from numba import njit, prange
import os

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
        self.cerjan = np.ones((self.c.nz,self.c.nx))  
        self.bord = np.zeros(self.c.nabc)
        self.sb = 1.5 * self.c.nabc
        #self.P = np.zeros((self.c.nz,self.c.nx,self.c.nt),dtype=np.float32)
        self.snap=np.zeros((self.c.nz,self.c.nx,500))
        self.CSG=np.zeros((self.c.nt,len(self.m.rx)))
        self.CRG=np.zeros((self.c.nt,len(self.m.rx)))
        self.upre =  np.zeros((self.c.nz,self.c.nx),dtype=np.float32)
        self.ufut =  np.zeros((self.c.nz,self.c.nx),dtype=np.float32)
        self.upas =  np.zeros((self.c.nz,self.c.nx),dtype=np.float32)
        self.c.factor= 0.0015
        self.damp = 0
        self.damp_x= np.zeros(self.c.nx)
        self.damp_z= np.zeros(self.c.nz)
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


    def make_cerjan(self):
        
        nabc = self.c.nabc
        nz = self.c.nz
        nx = self.c.nx

        # Amortecimento vertical: topo e fundo
        for i in range(nz):

            if i < nabc:
                d = nabc - i
                self.damp_z[i] = np.exp(-(self.c.factor * d) ** 2)

            elif i >= nz - nabc:
                d = i - (nz - nabc - 1)
                self.damp_z[i] = np.exp(-(self.c.factor * d) ** 2)

            else:
                self.damp_z[i] = 1.0

        # Amortecimento horizontal: esquerda e direita
        for j in range(nx):

            if j < nabc:
                d = nabc - j
                self.damp_x[j] = np.exp(-(self.c.factor * d) ** 2)

            elif j >= nx - nabc:
                d = j - (nx - nabc - 1)
                self.damp_x[j] = np.exp(-(self.c.factor * d) ** 2)

            else:
                self.damp_x[j] = 1.0

        print(self.damp_x)
    @measure_runtime
    def eq2DS(self):

        #dlay= 50

        

        print(self.c.s.f0,)

        ricker1 = wavelet(self.c.s.f0,self.m.time)
        #source2 = np.zeros(self.c.nt)

       #wave= wavelet(self.c.s.f0,np.arange(0,(self.c.nt-dlay)*self.m.dt,self.m.dt))

        #source2[dlay:]= wave

        
        
        dh2 = self.m.dh * self.m.dh
        
        #cte = (self.m.model * self.m.dt)**2
        cte = (self.m.marmo * self.m.dt)**2
        s=0
        for sh in range(16):

            self.upas[:, :] = 0.0
            self.upre[:, :] = 0.0
            self.ufut[:, :] = 0.0
            self.CSG[:, :] = 0.0
            sz = int(self.m.sz[sh] / self.m.dh) + self.m.nabc
            sx = int(self.m.sx[sh] / self.m.dh) + self.m.nabc

            for t in range(1, self.c.nt-1):
                
                
                # fonte
                self.upre[sz, sx] += (self.m.dt**2 / dh2) * ricker1[t]
                
                #self.P[20, self.m.sx+40, t] += source2[t] / dh2
                
                


                self.ufut,self.upre,self.upas = laplacian2d(
                    self.upre, self.c.nz, self.c.nx, dh2, self.upas,self.ufut,self.damp_x,self.damp_z,cte
                )


                #self.P[:, :, t+1] = cte * laplacian + 2*self.P[:, :, t] - self.P[:, :, t-1]
                #self.ufut[:,:] = cte * laplacian + 2*self.upre - self.upas
                # salvar snapshots a um certo passo de tempo

                
                
                
                # if t%4==0 and s<500:
                    
                #     self.snap[:,:,s] = self.upre[:,:]
                #     s += 1

                for j in range(len(self.m.rx)):
                    iz = int(self.m.rz[j] / self.m.dh) + self.m.nabc
                    ix = int(self.m.rx[j] / self.m.dh) + self.m.nabc

                    self.CSG[t, j] = self.upre[iz, ix]
                
                self.upas, self.upre, self.ufut = self.upre, self.ufut, self.upas
                        
            self.CSG.astype('float32').flatten(order='F').tofile(f'./Sismogram/Sismogram_P_NT2000_R170_SHOT{sh+1}.bin')



    def eq2DR(self):

        ns = 170      # número de tiros
        nr = 36       # número de receptores
        nt = self.c.nt

        os.makedirs("./CRG", exist_ok=True)

        ricker1 = wavelet(self.c.s.f0, self.m.time)

        dh2 = self.m.dh * self.m.dh
        cte = (self.m.marmo * self.m.dt) ** 2

        # posições dos receptores já convertidas para índice da malha
        rz_all = (self.m.rz[:nr] / self.m.dh).astype(int) + self.m.nabc
        rx_all = (self.m.rx[:nr] / self.m.dh).astype(int) + self.m.nabc

        # cria os arquivos CRG direto no disco
        crg_files = []

        for rec in range(nr):

            fname = f"./CRG/CRG_REC{rec+1}_NT{nt}_NS{ns}.bin"

            crg = np.memmap(
                fname,
                dtype="float32",
                mode="w+",
                shape=(nt, ns),
                order="F"
            )

            crg[:, :] = 0.0
            crg_files.append(crg)

        # agora roda cada tiro uma única vez
        for sh in range(ns):

            print(f"Rodando tiro {sh+1}/{ns}")

            self.upas[:, :] = 0.0
            self.upre[:, :] = 0.0
            self.ufut[:, :] = 0.0

            sz = int(self.m.sz[sh] / self.m.dh) + self.m.nabc
            sx = int(self.m.sx[sh] / self.m.dh) + self.m.nabc

            for t in range(1, nt - 1):

                # fonte
                self.upre[sz, sx] += ricker1[t]

                self.ufut, self.upre, self.upas = laplacian2d(
                    self.upre,
                    self.c.nz,
                    self.c.nx,
                    dh2,
                    self.upas,
                    self.ufut,
                    self.damp_x,
                    self.damp_z,
                    cte
                )

                # salva direto no CRG:
                # cada receptor recebe uma coluna do tiro sh
                for rec in range(nr):

                    rz = rz_all[rec]
                    rx = rx_all[rec]

                    crg_files[rec][t, sh] = self.upre[rz, rx]

                self.upas, self.upre, self.ufut = self.upre, self.ufut, self.upas

        # garante que tudo foi salvo no disco
        for rec in range(nr):
            crg_files[rec].flush()

        print("CRGs salvos com sucesso!")


    def plot2D(self):
        
        # isnap = 150

        # snap_plot = self.snap[:, :, isnap]
        # print("min snap:", np.min(snap_plot))
        # print("max snap:", np.max(snap_plot))
        # print("maior abs:", np.max(np.abs(snap_plot)))
        # print("tem NaN?", np.isnan(snap_plot).any())

        # abs_snap = np.abs(snap_plot)
        # vmax = np.percentile(abs_snap, 99)
        # vmin = -vmax

        # plt.figure(figsize=(10, 5))

        # plt.imshow(
        #     snap_plot,
        #     cmap='gray',
        #     aspect='auto',
        #     extent=[0, self.c.nx*self.m.dh, self.c.nz*self.m.dh, 0],
        #     vmax=vmax,
        #     vmin=vmin
        # )

        # plt.colorbar(label="Amplitude")
        # plt.xlabel("x (m)")
        # plt.ylabel("z (m)")
        # plt.title(f"Snapshot {isnap}")
        # plt.show()

        arquivo = f"Sismogram/Sismogram_P_NT2000_R170_SHOT15.bin"

        sismo = np.fromfile(arquivo, dtype="float32")

        sismo = sismo.reshape((self.c.nt, len(self.m.rx)), order="F")

        vmax = np.std(sismo)
        vmin = -vmax

        plt.figure(figsize=(10, 6))

        plt.imshow(
            sismo,
            cmap="gray",
            aspect="auto",
            extent=[
                self.m.rx[0],
                self.m.rx[-1],
                self.c.nt * self.m.dt,
                0
            ],
            vmin=vmin,
            vmax=vmax
        )

        plt.colorbar(label="Amplitude")
        plt.xlabel("Receiver position x (m)")
        plt.ylabel("Time (s)")
        plt.title("Seismogram")
        plt.show()
        
    def animation2D(self):
            
        from matplotlib.animation import FuncAnimation

        fig,ax = plt.subplots()

        ax.imshow(self.m.model,cmap='gray',aspect='auto', extent=[0, self.c.nx*self.m.dh, self.c.nz*self.m.dh, 0],alpha=0.5)
        wave = ax.imshow(self.snap[:, :, 0], cmap="gray", aspect="auto",extent=[0, self.c.nx*self.m.dh, self.c.nz*self.m.dh, 0],alpha=0.7)


        ax.set_xlabel("x (m)")
        ax.set_ylabel("twt (s)")
        ax.set_title("Snapshot")


        def atualizar(frame):
            
            abs_snap = np.abs(self.snap)
            vmax = np.percentile(abs_snap, 99)
            vmin = -vmax
            wave.set_data(self.snap[:,:,frame])
            wave.set_clim(vmin, vmax)
            ax.set_title(f"time = {frame*self.m.dt:.3f} s")
            
            return wave,


        ani = FuncAnimation(fig, atualizar, frames=500, interval=10)
        #ani.save('monda2d.gif',writer='pilow',fps=30)
        plt.show()

@njit(parallel=True)
def laplace1D(P,t,dz,nz):
    
    lap = np.zeros(nz)
    for n in range(2, nz-2):
        
        lap[n] = (-P[n+2, t] + 16*P[n+1, t] - 30*P[n, t] + 16*P[n-1, t] - P[n-2, t])  / (12 * dz**2)
    
    return lap


@njit(parallel=True)
def laplacian2d(upre, nzz, nxx, dh2,upas,ufut,damp_x,damp_z,cte):

    lap = np.zeros((nzz, nxx), dtype=np.float32)

    inv_dh2 = 1.0 / (5040.0 * dh2)

    for i in prange(4, nzz - 4):
        for j in range(4, nxx - 4):

            d2x = (
                -9.0 * upre[i-4, j]
                + 128.0 * upre[i-3, j]
                - 1008.0 * upre[i-2, j]
                + 8064.0 * upre[i-1, j]
                - 14350.0 * upre[i, j]
                + 8064.0 * upre[i+1, j]
                - 1008.0 * upre[i+2, j]
                + 128.0 * upre[i+3, j]
                - 9.0 * upre[i+4, j]
            )

            d2z = (
                -9.0 * upre[i, j-4]
                + 128.0 * upre[i, j-3]
                - 1008.0 * upre[i, j-2]
                + 8064.0 * upre[i, j-1]
                - 14350.0 * upre[i, j]
                + 8064.0 * upre[i, j+1]
                - 1008.0 * upre[i, j+2]
                + 128.0 * upre[i, j+3]
                - 9.0 * upre[i, j+4]
            )

            
            lap[i, j] = (d2x + d2z) * inv_dh2

            damp = damp_z[i] * damp_x[j]

            ufut[i, j] = (
                cte[i, j] * lap[i, j]
                + 2.0 * upre[i, j]
                - upas[i, j]
            ) 
           
    for i in prange(nzz):
        for j in range(nxx):

            damp = damp_z[i] * damp_x[j]

            ufut[i, j] *= damp
            upre[i, j] *= damp
            upas[i, j] *= damp

    

    return ufut,upre,upas

def wavelet(freq,t):
  f_corte = freq

  fc = f_corte / (3 * np.sqrt(np.pi))

  td = t - (0.5 * np.sqrt(np.pi) / fc)

  arg = np.pi * (np.pi * fc * td)**2

  return (1 - 2*arg) * np.exp(-arg)
