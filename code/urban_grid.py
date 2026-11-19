"""
Cross-range (Ny) grid-convergence and side-edge angular-spectrum check.
 - Rerun the identical urban block at Ny=192 (dy=0.625 m) and Ny=384
   (dy=0.3125 m) -> lateral Nyquist angles 15.5 deg and 32.2 deg.
 - Compare the y=0,z=15 path-loss curve (convergence in the observable).
 - From the fine (Ny=384) cross-planes, FFT the physical y-window and
   integrate the transverse angular power to quantify how much energy lies
   beyond the coarse-grid 15.5 deg lateral Nyquist limit (the side-edge
   diffraction content that the coarse grid cannot represent).
Only the receiver line and a few cross-planes are stored (Ny=384 full 3D
history would be ~20 GB).
"""
import time
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from kuttler_core3d import Kuttler3D
from urban_lib import f,eps_c,blds,Tx_y,Tx_z,sig,Lx,dx,Nx,X

def march(Ny,zr=15.,yr=0.,slice_x=(450,600,750,950)):
    Ly=120.;H=60.;Nz=4096;af=0.16
    t0=time.time()
    S=Kuttler3D(f,Ly=Ly,Ny=Ny,H=H,Nz=Nz,abs_frac=af)
    S.set_ground(eps_c=eps_c,pol='H')
    y=S.y;z=S.z
    Y,Z=np.meshgrid(y,z,indexing='ij')
    u=np.exp(-(Y-Tx_y)**2/(2*sig**2))*np.exp(-(Z-Tx_z)**2/(2*sig**2))
    iy=np.argmin(np.abs(y-yr));iz=np.argmin(np.abs(z-zr))
    line=np.zeros(Nx);n0=np.abs(u[iy,np.argmin(np.abs(z-Tx_z))])
    slices={};sidx={int(x/dx) for x in slice_x}
    for i in range(Nx):
        line[i]=20*np.log10(n0/(np.abs(u[iy,iz])+1e-12))
        if i in sidx: slices[i]=u.copy()
        xv=i*dx
        for x0,x1,y1,y2,h in blds:
            if x0<=xv<x0+dx: u[(Y>=y1)&(Y<=y2)&(Z<=h)]=0
        u=S.step(u,dx)
    return line,slices,y,z,time.time()-t0,Ly

print('marching Ny=192 ...');l192,sl192,y192,z192,t192,Ly=march(192)
print('marching Ny=384 ...');l384,sl384,y384,z384,t384,_=march(384)
print('elapsed Ny192=%.1fs  Ny384=%.1fs'%(t192,t384))

# ---- convergence of the observable (exclude in-building + near edges) ----
valid=X>300
for x0,x1,y1,y2,h in blds:
    if y1<=0<=y2 and h>=15: valid[(X>=x0-2)&(X<=x1+5)]=False
dd=np.abs(l192-l384)[valid]
print('curve Ny192 vs Ny384: mean=%.3f RMS=%.3f max=%.3f dB'%(dd.mean(),np.sqrt((dd**2).mean()),dd.max()))

# ---- lateral angular power spectrum from the FINE grid ----
def ang_spec(u,y,z,Ly,af=0.16):
    # physical y-window (drop the Gaussian apodized edges)
    yphy=Ly*(0.5-af)
    my=np.abs(y)<=yphy
    mz=(z>=2)&(z<=40)                       # physical z 2..40 m
    n=my.sum();dy=np.mean(np.diff(y))
    U=np.fft.fft(u[np.ix_(my,mz)]*np.hanning(n)[:,None],axis=0)
    ky=np.fft.fftfreq(n,dy)*2*np.pi
    Pky=np.abs(U)**2
    Pky=Pky.sum(axis=1)
    # fold +/- ky into one-sided |ky| power
    kypos=np.fft.rfftfreq(n,dy)*2*np.pi
    Ppos=np.zeros(len(kypos));Ppos[0]=Pky[0]
    for j in range(1,len(kypos)):
        Ppos[j]=Pky[j]+(Pky[n-j] if n-j!=j else 0)
    return kypos,Ppos
k0=2*np.pi/ (3e8/f)
theta_c=15.47; ky_c=k0*np.sin(theta_c*np.pi/180)
print('\nTransverse energy beyond coarse Nyquist 15.5 deg (from Ny=384):')
fracs=[]
for ix in sorted(sl384):
    ky,P=ang_spec(sl384[ix],y384,z384,Ly)
    tot=P.sum();hi=P[ky>ky_c].sum()
    fracs.append(hi/tot)
    print('  x=%4d m: fraction = %.4f%%'%(int(ix*dx),100*hi/tot))
print('mean fraction beyond 15.5 deg = %.4f%%'%(100*np.mean(fracs)))

# representative angle spectrum (average of slices)
allP=None
for ix in sorted(sl384):
    ky,P=ang_spec(sl384[ix],y384,z384,Ly)
    ang=np.degrees(np.arcsin(np.clip(ky/k0,-1,1)))
    allP=P if allP is None else allP+P
allP/=len(sl384)

fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].plot(X,l192,'b-',lw=2,label='Ny=192, Δy=0.625 m (15.5°)')
ax[0].plot(X,l384,'r--',lw=1.6,label='Ny=384, Δy=0.3125 m (32.2°)')
for x0,x1,y1,y2,h in blds:
    if y1<=0<=y2 and h>=15: ax[0].axvspan(x0,x1,color='gray',alpha=0.2)
ax[0].set_ylim(0,70);ax[0].set_xlabel('Range x (m)');ax[0].set_ylabel('Path loss (dB)')
ax[0].set_title('(a) Cross-range grid convergence (y=0,z=15)');ax[0].legend();ax[0].grid(alpha=0.3)
ax[1].plot(ang,allP/allP.max(),'k-',lw=2)
ax[1].axvline(15.47,color='b',ls='--',label='Ny=192 limit 15.5°')
ax[1].axvline(32.2,color='r',ls='--',label='Ny=384 limit 32.2°')
ax[1].set_yscale('log');ax[1].set_xlim(0,60);ax[1].set_ylim(1e-6,1.1)
ax[1].set_xlabel('Transverse angle (deg)');ax[1].set_ylabel('Normalized angular power')
ax[1].set_title('(b) Side-edge angular spectrum (behind buildings)');ax[1].legend();ax[1].grid(alpha=0.3,which='both')
plt.tight_layout();plt.savefig('fig_urban_grid_convergence.png',dpi=150);plt.close()
print('saved fig_urban_grid_convergence.png')
np.savez('urban_grid.npz',l192=l192,l384=l384,X=X,t192=t192,t384=t384,fracs=fracs)
