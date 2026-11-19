"""
Verified 3D Kuttler single-field SSFT-PE (cross-plane y-z), extending the
2D Central-Difference DMFT.
  - y direction: periodic FFT with Gaussian absorption at both ends.
  - z direction: half-grid Central-Difference DMFT at the ground z=0
    (Leontovich u'(0)+a u(0)=0, H-pol a=-j k0 sqrt(eps_c-1); PEC |a|->inf),
    Gaussian absorption at the top.
  Ground mode:
    'kuttler' : CD-DMFT at z=0 (impedance / PEC).
    'free'    : periodic FFT in z with absorption at both ends (free-space check).
"""
import numpy as np
from scipy.fft import dst,idst,fft,ifft

C0=3e8
class Kuttler3D:
    def __init__(self,f=900e6,Ly=200.,Ny=256,H=120.,Nz=4096,abs_frac=0.18):
        self.f=f;self.lam=C0/f;self.k0=2*np.pi/self.lam
        self.Ly=Ly;self.Ny=Ny;self.dy=Ly/Ny
        self.y=np.arange(Ny)*self.dy-Ly/2
        self.ky=2*np.pi*np.fft.fftfreq(Ny,self.dy)
        self.H=H;self.Nz=Nz;self.dz=H/Nz
        self.z=(np.arange(Nz)+0.5)*self.dz
        kz=np.pi*np.arange(1,Nz)/(Nz*self.dz)
        self.kz=kz
        # z absorption (top) over the Nz-1 interior DST points
        self.abz=np.ones(Nz-1);w=int((Nz-1)*abs_frac)
        for i in range(w):self.abz[Nz-2-i]=np.exp(-((w-i)/w)**2*6)
        # y absorption (both ends), spatial domain
        self.aby=np.ones(Ny);wy=int(Ny*abs_frac)
        for i in range(wy):
            v=np.exp(-((wy-i)/wy)**2*6);self.aby[i]=v;self.aby[Ny-1-i]=v
        # 2D propagation operator kx(ky,kz)
        K2=self.k0**2-self.ky[:,None]**2-kz[None,:]**2
        self.kx=np.zeros_like(K2,dtype=complex);mm=K2>=0
        self.kx[mm]=np.sqrt(K2[mm]);self.kx[~mm]=-1j*np.sqrt(-K2[~mm])
        # free-mode z periodic grid
        self.zp=np.arange(Nz)*self.dz
        self.kzp=2*np.pi*np.fft.fftfreq(Nz,self.dz)
        K2f=self.k0**2-self.ky[:,None]**2-self.kzp[None,:]**2
        self.kxf=np.zeros_like(K2f,dtype=complex);mf=K2f>=0
        self.kxf[mf]=np.sqrt(K2f[mf]);self.kxf[~mf]=-1j*np.sqrt(-K2f[~mf])
        self.abzf=np.ones(Nz);w=int(Nz*abs_frac)
        for i in range(w):
            v=np.exp(-((w-i)/w)**2*6);self.abzf[i]=v;self.abzf[Nz-1-i]=v
        self.mode='free';self.r=None
    def set_ground(self,eps_c=None,pec=False,pol='H'):
        # Leontovich u'(0)+alpha u=0, alpha=-j k0 Delta
        #  TE/H-pol: Delta_H=sqrt(eps_c-1);  TM/V-pol: Delta_V=sqrt(eps_c-1)/eps_c
        self.mode='kuttler';self.pol=pol
        if pec:alpha=1j*self.k0*1e8
        else:
            Delta=np.sqrt(eps_c-1.0) if pol=='H' else np.sqrt(eps_c-1.0)/eps_c
            alpha=-1j*self.k0*Delta
        self.alpha=alpha;self.r=(1-alpha*self.dz/2)/(1+alpha*self.dz/2)
    def set_free(self):
        self.mode='free'
    def _fwd_z(self,U):
        return U[:,1:self.Nz]-self.r*U[:,0:self.Nz-1]   # [Ny, Nz-1] w_1..w_{N-1}
    def _inv_z(self,win):
        N=self.Nz;r=self.r;Ny=self.Ny
        u=np.zeros((Ny,N),dtype=complex)
        if abs(r)<1:
            up=np.zeros((Ny,N),dtype=complex)
            for k in range(1,N):up[:,k]=win[:,k-1]+r*up[:,k-1]
            A=-up[:,N-1]/r**(N-1)
            u=up+A[:,None]*r**np.arange(N)
        else:
            uN=np.zeros(Ny,dtype=complex)
            for k in range(N,0,-1):
                wk=0 if k==N else win[:,k-1]
                uN=(uN-wk)/r;u[:,k-1]=uN
        return u
    def step(self,u,dx):
        if self.mode=='kuttler':
            U=fft(u,axis=0)
            win=self._fwd_z(U)
            win=dst(win,type=1,axis=1,norm='ortho')
            win*=np.exp(-1j*self.kx*dx)
            win=idst(win,type=1,axis=1,norm='ortho')*self.abz[None,:]
            u=self._inv_z(win)
            u=ifft(u,axis=0)
            u*=self.aby[:,None]
        else:
            U=fft(u,axis=0)
            U=np.fft.fft(U,axis=1)
            U*=np.exp(-1j*self.kxf*dx)
            u=np.fft.ifft(U,axis=1)
            u=ifft(u,axis=0)
            u*=self.aby[:,None]*self.abzf[None,:]
        return u
    def gaussian_source(self,y0,zc,sigy,sigz,angle_y_deg=0,angle_z_deg=0):
        sy=self.k0*np.sin(angle_y_deg*np.pi/180);sz=self.k0*np.sin(angle_z_deg*np.pi/180)
        Y,Z=np.meshgrid(self.y,self.z,indexing='ij')
        return np.exp(-(Y-y0)**2/(2*sigy**2))*np.exp(-(Z-zc)**2/(2*sigz**2))*np.exp(1j*sy*Y+1j*sz*Z)
    def gaussian_source_free(self,y0,zc,sigy,sigz):
        Y,Z=np.meshgrid(self.y,self.zp,indexing='ij')
        return np.exp(-(Y-y0)**2/(2*sigy**2))*np.exp(-(Z-zc)**2/(2*sigz**2))
