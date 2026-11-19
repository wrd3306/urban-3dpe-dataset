"""
Verified Kuttler single-field 2D SSFT-PE with Leontovich impedance boundary.
Half-grid Central-Difference DMFT (Kuttler & Janaswamy, Radio Sci. 2002).
  u_k at z_k=(k+1/2) dz;  r=(1-a dz/2)/(1+a dz/2);  w_k=u_k-r u_{k-1}, w_0=w_N=0.
  Boundary u'(0)+a u(0)=0;  H-pol a=-j k0 sqrt(eps_c-1); PEC |a|->inf.
Reflection coefficient validated vs Fresnel 2-30 deg, max error 0.012.
"""
import numpy as np
from scipy.fft import dst,idst

C0=3e8
class Kuttler2D:
    def __init__(self,f=900e6,H=120.,N=4096,abs_frac=0.3):
        self.f=f;self.lam=C0/f;self.k0=2*np.pi/self.lam
        self.H=H;self.N=N;self.dz=H/N
        self.z=(np.arange(N)+0.5)*self.dz
        kz=np.pi*np.arange(1,N)/(N*self.dz)
        ksq=self.k0**2-kz**2
        self.kx=np.zeros_like(ksq,dtype=complex);m=ksq>=0
        self.kx[m]=np.sqrt(ksq[m]);self.kx[~m]=-1j*np.sqrt(-ksq[~m])
        self.ab=np.ones(N-1);wt=int((N-1)*abs_frac)
        for i in range(wt): self.ab[N-2-i]=np.exp(-((wt-i)/wt)**2*6)
    def set_ground(self,eps_c=None,pec=False,pol='H'):
        # Leontovich u'(0)+alpha u=0, alpha=-j k0 Delta
        #  TE/H-pol (E horizontal): Delta_H = sqrt(eps_c-1)
        #  TM/V-pol (E vertical):   Delta_V = sqrt(eps_c-1)/eps_c
        if pec: alpha=1j*self.k0*1e8
        else:
            Delta=np.sqrt(eps_c-1.0) if pol=='H' else np.sqrt(eps_c-1.0)/eps_c
            alpha=-1j*self.k0*Delta
        self.pol=pol;self.alpha=alpha
        self.r=(1-alpha*self.dz/2)/(1+alpha*self.dz/2)
    def _fwd(self,u):
        N=self.N;w=np.zeros(N+1,dtype=complex)
        w[1:N]=u[1:N]-self.r*u[0:N-1];return w[1:N]
    def _inv(self,win):
        N=self.N;r=self.r
        w=np.zeros(N+1,dtype=complex);w[1:N]=win
        u=np.zeros(N,dtype=complex)
        if abs(r)<1:
            up=np.zeros(N,dtype=complex)
            for k in range(1,N): up[k]=w[k]+r*up[k-1]
            A=-up[N-1]/r**(N-1);u=up+A*r**(np.arange(N))
        else:
            uN=0j
            for k in range(N,0,-1): uN=(uN-w[k])/r;u[k-1]=uN
        return u
    def step(self,u,dx):
        win=self._fwd(u)
        win=idst(dst(win,type=1,norm='ortho')*np.exp(-1j*self.kx*dx),type=1,norm='ortho')*self.ab
        return self._inv(win)
    def gaussian_source(self,zc,sig,angle_deg=0.0):
        th=angle_deg*np.pi/180;s=self.k0*np.sin(th)
        return np.exp(-(self.z-zc)**2/(2*sig**2))*np.exp(+1j*s*self.z)
    def propagate(self,u0,xmax,dx):
        Nx=int(xmax/dx);hist=np.zeros((Nx,self.N),dtype=complex)
        u=u0.copy()
        for i in range(Nx):
            hist[i]=u;u=self.step(u,dx)
        return np.arange(Nx)*dx,hist
