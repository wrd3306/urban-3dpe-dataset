"""
Reusable 3D urban micro-cell SSFT-PE runner shared by the V-pol study, the
cross-range (Ny) grid-convergence study, the side-edge angular-spectrum check
and the multi-receiver statistics. Geometry/grid are identical to urban3d_pe.py
so all results are directly comparable.

Buildings are thin absorbing knife screens: only the front face at x0 is
nulled in a single range step (a finite-thickness block would be counted as
two diffractions); the listed 3 m thickness is for visualization only.
"""
import time
import numpy as np
from kuttler_core3d import Kuttler3D

f=900e6;c=3e8;lam=c/f;k0=2*np.pi/lam
eps_c=5.0-1j*0.01/(2*np.pi*f*8.854e-12)

def sc(xc,t=3.):return (xc-t/2,xc+t/2)
# (x0,x1, y1,y2, h) ; thickness t=3 m visualization only
blds=[(*sc(205),-30,30,10),(*sc(375),-30,5,25),(*sc(545),-5,30,16),
      (*sc(705),-30,30,12),(*sc(875),-20,30,30)]

Tx_y=0.;Tx_z=20.;sig=3.
Lx=1000.;dx=1.25;Nx=int(Lx/dx)
X=np.arange(Nx)*dx

def run_urban(Ny=192, pol='H', pec=False, Ly=120., H=60., Nz=4096,
              abs_frac=0.16, verbose=True):
    """March the 3D urban block; return hist[Nx,Ny,Nz], y, z, elapsed."""
    t0=time.time()
    S=Kuttler3D(f,Ly=Ly,Ny=Ny,H=H,Nz=Nz,abs_frac=abs_frac)
    if pec: S.set_ground(pec=True)
    else:   S.set_ground(eps_c=eps_c,pol=pol)
    y=S.y;z=S.z
    Y,Z=np.meshgrid(y,z,indexing='ij')
    u=np.exp(-(Y-Tx_y)**2/(2*sig**2))*np.exp(-(Z-Tx_z)**2/(2*sig**2))
    hist=np.zeros((Nx,S.Ny,S.Nz),dtype=complex)
    for i in range(Nx):
        hist[i]=u
        xv=i*dx
        for x0,x1,y1,y2,h in blds:
            if x0<=xv<x0+dx:                      # front-face single-plane null
                u[(Y>=y1)&(Y<=y2)&(Z<=h)]=0
        u=S.step(u,dx)
        if verbose and (i+1)%100==0:
            print('  %s Ny=%d step %d/%d (%.0f s)'%(pol,Ny,i+1,Nx,time.time()-t0))
    return hist,y,z,time.time()-t0

def path_loss(hist,y,z,zr=15.,yr=0.):
    iy=np.argmin(np.abs(y-yr));iz=np.argmin(np.abs(z-zr))
    n0=np.abs(hist[0,iy,np.argmin(np.abs(z-Tx_z))])
    return 20*np.log10(n0/(np.abs(hist[:,iy,iz])+1e-12))
