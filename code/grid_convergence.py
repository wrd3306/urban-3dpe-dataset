import numpy as np
from scipy.fft import dst,idst
from scipy.signal import lfilter
f=900e6;c=3e8;lam=c/f;k0=2*np.pi/lam
eps_c=5.0-1j*0.01/(2*np.pi*f*8.854e-12)
Delta=np.sqrt(eps_c-1.0)
def measure(N,H=120.,theta=15*np.pi/180,sig=10.,zc=70.,xobs=420.,dx=0.25):
    dz=H/N;z=(np.arange(N)+1)*dz
    kz=np.pi*np.arange(1,N+1)/((N+1)*dz)
    ksq=k0**2-kz**2;kx=np.zeros_like(ksq,dtype=complex);m=ksq>=0
    kx[m]=np.sqrt(ksq[m]);kx[~m]=-1j*np.sqrt(-ksq[~m]);prop=np.exp(-1j*kx*dx)
    ab=np.ones(N);wt=int(N*0.25)
    for i in range(wt): ab[N-1-i]=np.exp(-((wt-i)/wt)**2*6)
    s=k0*np.sin(theta)
    def run(alpha):
        r=1/(1+alpha*dz)
        u=np.exp(-(z-zc)**2/(2*sig**2))*np.exp(-1j*s*z)
        for _ in range(int(xobs/dx)):
            u0=r*u[0];w=np.empty(N,dtype=complex)
            w[0]=u[0]-r*u0;w[1:]=u[1:]-r*u[:-1]
            w=idst(dst(w,type=1,norm='ortho')*prop,type=1,norm='ortho')
            up=lfilter([1.],[1,-r],w);Cc=r*up[0]/(1-r*r)
            u=up+Cc*r**(np.arange(1,N+1));u*=ab
        return u
    xh=zc/np.tan(theta);zc2=(xobs-xh)*np.tan(theta)
    def amp(u):
        win=np.abs(z-zc2)<12
        Z=z[win];A=np.stack([np.exp(-(Z-zc2)**2/(2*sig**2))*np.exp(1j*s*Z)],axis=1)
        coef,_,_,_=np.linalg.lstsq(A,u[win],rcond=None);return coef[0]
    gC=amp(run(-1j*k0*Delta))   # H-pol alpha
    gP=amp(run(1j*k0*1e6))      # PEC
    return gC/gP,dz
gf=(np.sin(np.pi/12)-np.sqrt(eps_c-1))/(np.sin(np.pi/12)+np.sqrt(eps_c-1))
print('H-pol Fresnel @15deg: |G|=%.4f arg=%.1f deg'%(abs(gf),np.degrees(np.angle(gf))))
for N in [2048,4096,8192,16384]:
    g,dz=measure(N)
    print(f'N={N:6d} dz={dz:.4f}m k0dz={k0*dz:.3f} |a|dz={k0*abs(Delta)*dz:.3f} | |G|={abs(g):.4f} arg={np.degrees(np.angle(g)):7.1f} err={abs(g-gf):.4f}')
