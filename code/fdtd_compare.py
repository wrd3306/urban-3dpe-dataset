"""
Independent full-wave validation: 2D TM-Ez FDTD reference vs 2D SSFT-PE and
analytic solutions in three small canonical problems.
  free   : point-source cylindrical spreading vs Hankel H0^(2)
  pec    : PEC ground two-ray interference vs H0(r1)-H0(r2)
  screen : PEC half-plane shadow vs the soft-screen SSFT-PE (Fresnel knife)
The FDTD uses a continuous-wave point source at (xs,zs); the PE is given the
analytic point-source section (Hankel, plus the image for the PEC ground) on
its initial plane, so both methods start from the same omnidirectional source.
The PE source is at x=0 and the FDTD source at x=4; cuts are compared on
source-relative range, and as relative level (dB) to a common reference.
"""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.special import hankel2
from scipy.integrate import quad
from kuttler_core import Kuttler2D
C0=3e8;f=900e6;lam=C0/f;k0=2*np.pi/lam
DD=0.015;DXS=4.0;FS='_fine'   # fine FDTD grid lambda/22

def pe_free(Dz,Dx,zs,dx=0.1,dz=0.015,abf=0.32,screen=None):
    N=int(Dz/dz);Nx=int(Dx/dx);z=np.arange(N)*dz
    kz=2*np.pi*np.fft.fftfreq(N,dz);ksq=k0**2-kz**2
    kx=np.zeros(N,complex);m=ksq>=0;kx[m]=np.sqrt(ksq[m]);kx[~m]=-1j*np.sqrt(-ksq[~m])
    ab=np.ones(N);w=int(N*abf)
    for i in range(w):ab[i]=np.exp(-((w-i)/w)**2*8);ab[N-1-i]=ab[i]
    u=hankel2(0,k0*np.maximum(np.abs(z-zs),0.08));u=u/np.max(np.abs(u))
    H=np.zeros((Nx,N),complex);xe,h=(screen if screen else(None,None))
    for i in range(Nx):
        H[i]=u
        if screen is not None and i==int(xe/dx):u[z<h]=0
        U=np.fft.fft(u);U*=np.exp(-1j*kx*dx);u=np.fft.ifft(U)*ab
    return np.arange(Nx)*dx,z,H

def pe_pec(Dz,Dx,zs,dx=0.1,dz=0.015,abf=0.3):
    S=Kuttler2D(f,H=Dz,N=int(Dz/dz),abs_frac=abf);S.set_ground(pec=True)
    u=(hankel2(0,k0*np.maximum(np.abs(S.z-zs),0.08))
       -hankel2(0,k0*np.maximum(np.abs(S.z+zs),0.08)));u=u/np.max(np.abs(u))
    x,H=S.propagate(u,Dx,dx);return x,S.z,H

fd={k:np.load('fdtd_%s%s.npz'%(k,FS)) for k in ('free','pec','screen')}
fig,ax=plt.subplots(3,2,figsize=(13,11))

# ---- (1) free cylindrical spreading, horizontal line through source ----
xf,zf,Ef,sf=fd['free'].values() if False else (fd['free']['x'],fd['free']['z'],fd['free']['Ec'],fd['free']['src'])
ixs,jzs=sf;xs_,zs_=xf[ixs],zf[jzs]
iz=np.argmin(np.abs(zf-zs_));xi=xf[ixs:];rho=xi-xs_
valid=rho>0.5
Efd=Ef[ixs:,iz];Efd=Efd/np.abs(Efd[np.argmin(np.abs(xi-12))])
Han=hankel2(0,k0*np.maximum(rho,0.08));Han=Han/np.abs(Han[np.argmin(np.abs(xi-12))])
zPEc=40.  # PE free-space domain centres the source (vertical offset immaterial)
xp,zp,Hp=pe_free(80,22,zPEc);izp=np.argmin(np.abs(zp-zPEc))
Epe=Hp[:,izp];Epe=Epe/np.abs(Epe[np.argmin(np.abs(xp-8))])
ax[0,0].pcolormesh(xf,zf,20*np.log10(np.abs(Ef.T)+1e-9),cmap='jet',vmin=-55,vmax=20,shading='auto')
ax[0,0].plot(xs_,zs_,'w*',ms=12);ax[0,0].set_title('(a) FDTD free-space $E_z$');ax[0,0].set_xlabel('x (m)');ax[0,0].set_ylabel('z (m)')
mr=(xi>=12)&(xi<=22)
ax[0,1].plot(xi[valid],20*np.log10(np.abs(Efd[valid])),'r-',lw=2.2,label='FDTD')
ax[0,1].plot(xp+DXS,20*np.log10(np.abs(Epe)),'b--',lw=1.6,label='SSFT-PE')
ax[0,1].plot(xi[valid],20*np.log10(np.abs(Han[valid])),'o',color='k',mfc='none',ms=5,label='Hankel $H_0^{(2)}$')
ax[0,1].set_xlim(4.5,24.8);ax[0,1].set_ylim(-5,13)
ax[0,1].set_title('(b) On-axis cylindrical spreading');ax[0,1].set_xlabel('x (m)');ax[0,1].set_ylabel('relative level (dB)')
ax[0,1].legend();ax[0,1].grid(alpha=0.3)
mrp=(xp>=8)&(xp<=18)
Han_pe=hankel2(0,k0*np.maximum(xp,0.08));Han_pe=Han_pe/np.abs(Han_pe[np.argmin(np.abs(xp-8))])
eFree=np.sqrt(np.mean((20*np.log10(np.abs(Epe[mrp]))-20*np.log10(np.abs(Han_pe[mrp])))**2))
mr2=(xi>=12)&(xi<=20)
eFreeFD=np.sqrt(np.mean((20*np.log10(np.abs(Efd[mr2])+1e-12)-20*np.log10(np.abs(Han[mr2])+1e-12))**2))

# ---- (2) PEC ground, horizontal line at z=3 (two-ray interference) ----
xp_,zp_,Ep,sp=fd['pec']['x'],fd['pec']['z'],fd['pec']['Ec'],fd['pec']['src']
ixs2,jzs2=sp;xs2,zs2=xp_[ixs2],zp_[jzs2]
zr=3.;izr=np.argmin(np.abs(zp_-zr))
xl=xp_[ixs2:];Gf=Ep[ixs:,izr] if False else Ep[ixs2:,izr]
r1=np.hypot(xl-xs2,zs2-zr);r2=np.hypot(xl-xs2,zs2+zr)
Gim=hankel2(0,k0*r1)-hankel2(0,k0*r2)
xx,zz,Hq=pe_pec(40,20,zs2);izp2=np.argmin(np.abs(zz-zr))
Gpe=Hq[:,izp2]
# align in dB using a long-range stretch (common slope), compare x=12..20
sel=(xl>=12)&(xl<=20);selp=(xx>=8)&(xx<=16)
dF=20*np.log10(np.abs(Gf[sel]));dI=20*np.log10(np.abs(Gim[sel]));dP0=20*np.log10(np.abs(Gpe[selp]))
dP=np.interp(xl[sel],xx[selp]+DXS,dP0)
dF-=dF.mean();dI-=dI.mean();dP-=dP.mean()
ax[1,0].pcolormesh(xp_,zp_,20*np.log10(np.abs(Ep.T)+1e-9),cmap='jet',vmin=-50,vmax=22,shading='auto')
ax[1,0].axhline(0,color='w',lw=2);ax[1,0].plot(xs2,zs2,'w*',ms=12);ax[1,0].axhline(zr,color='w',ls='--',lw=1)
ax[1,0].set_title('(c) FDTD PEC-ground $E_z$');ax[1,0].set_xlabel('x (m)');ax[1,0].set_ylabel('z (m)')
ax[1,1].plot(xl[sel],dF,'r-',lw=2.2,label='FDTD')
ax[1,1].plot(xl[sel],dP,'b--',lw=1.6,label='SSFT-PE (PEC)')
ax[1,1].plot(xl[sel],dI,'o',color='k',mfc='none',ms=5,label='Image $H_0(r_1)-H_0(r_2)$')
ax[1,1].set_title('(d) PEC two-ray interference at z=3 m');ax[1,1].set_xlabel('x (m)');ax[1,1].set_ylabel('relative level (dB, mean removed)')
ax[1,1].set_ylim(-27,8)
ax[1,1].legend();ax[1,1].grid(alpha=0.3)
ePec=np.sqrt(np.mean((dF-dI)**2))
ePecP=np.sqrt(np.mean((dP-dI)**2))
ePecFP=np.sqrt(np.mean((dF-dP)**2))

# ---- (3) PEC half-plane, diffraction loss behind screen ----
xs3,z3,E3,s3=fd['screen']['x'],fd['screen']['z'],fd['screen']['Ec'],fd['screen']['src']
ixs3,jzs3=s3;xsr,zsr=xs3[ixs3],z3[jzs3];xe,h=12.,8.
Xcut=26.;ic2=np.argmin(np.abs(xs3-Xcut))
rho=np.hypot(Xcut-xsr,z3-zsr);Gf3o=hankel2(0,k0*rho)
raw_fd=20*np.log10(np.abs(E3[ic2,:])+1e-9)-20*np.log10(np.abs(Gf3o)+1e-9)
# Normalise the soft-source gain in the geometric-optical bright zone
# (high z, line of sight clears the edge), where the loss must be 0 dB.
gb=(z3>13)&(z3<16.5)
dl_fd=raw_fd-np.median(raw_fd[gb])
# PE: source at x=0, screen xe=8 (8 m behind source), cut at x=22.
# Vertically shift the domain so the screen extends down into the bottom
# absorber while the aperture and source sit in the clean middle region.
zoff=18.0;xeP=xe-DXS;XcutP=Xcut-DXS
zsP=zsr+zoff;hP=h+zoff
xxs,za,Hs=pe_free(40,26,zsP,abf=0.16,screen=(xeP,hP))
xxf,zb,Hf=pe_free(40,26,zsP,abf=0.16)
zzs=za-zoff;zzf=zb-zoff   # physical (FDTD) z coordinates
ics=np.argmin(np.abs(xxs-XcutP));icf=np.argmin(np.abs(xxf-XcutP))
dl_pe=20*np.log10(np.abs(Hs[ics,:])+1e-9)-20*np.log10(np.abs(Hf[icf,:])+1e-9)
# Fresnel-Kirchhoff soft knife-edge (plane-wave/cylindrical, Fresnel integral)
def fresnel_loss(zr):
    from scipy.special import fresnel as fr
    d1=xeP; d2=XcutP-xeP
    z_los=(d2*zsr+d1*zr)/(d1+d2)     # line-of-sight height at the screen
    h_e=h-z_los                      # edge above LOS (>0 in shadow)
    v=h_e*np.sqrt((2/lam)*(1/d1+1/d2))
    S,C=fr(v)
    F=0.5*(1+1j)*((0.5-C)+1j*(0.5-S))
    return 20*np.log10(np.abs(F))
dl_fk=np.array([fresnel_loss(zr) for zr in zzs])
mk2=(z3>1)&(z3<15);mkp=(zzs>1)&(zzs<15)
ax[2,0].pcolormesh(xs3,z3,20*np.log10(np.abs(E3.T)+1e-9),cmap='jet',vmin=-50,vmax=22,shading='auto')
ax[2,0].plot([xe,xe],[0,h],'w-',lw=2.5);ax[2,0].plot(xsr,zsr,'w*',ms=12)
ax[2,0].set_title('(e) FDTD PEC half-plane (screen x=12, h=8)');ax[2,0].set_xlabel('x (m)');ax[2,0].set_ylabel('z (m)')
ax[2,1].plot(z3[mk2],dl_fd[mk2],'r-',lw=2.2,label='FDTD (full wave)')
ax[2,1].plot(zzs[mkp],dl_pe[mkp],'b--',lw=1.6,label='SSFT-PE (soft screen)')
ax[2,1].plot(zzs[mkp],dl_fk[mkp],'o',color='k',mfc='none',ms=5,label='Fresnel knife-edge')
ax[2,1].axhline(h,color='gray',ls=':');ax[2,1].set_title('(f) Diffraction loss at x=26 m')
ax[2,1].set_xlabel('z (m)');ax[2,1].set_ylabel('diffraction loss (dB)');ax[2,1].set_ylim(-32,6)
ax[2,1].legend();ax[2,1].grid(alpha=0.3)
eSc=np.sqrt(np.mean((dl_fd[mk2]-np.interp(z3[mk2],zzs,dl_pe))**2))
eScF=np.sqrt(np.mean((dl_fd[mk2]-np.interp(z3[mk2],zzs,dl_fk))**2))

plt.tight_layout();plt.savefig('fig_fdtd_validation.png',dpi=150);plt.close()
print('free  PE-Hankel axis RMS = %.3f dB ; FDTD-Hankel = %.3f'%(eFree,eFreeFD))
print('pec   FDTD-PE line RMS         = %.3f dB ; FDTD-image = %.3f ; PE-image = %.3f'%(ePecFP,ePec,ePecP))
seln=(xl>=12)&(xl<=17); selpn=(xx+DXS>=12)&(xx+DXS<=17)
dPn=np.interp(xl[seln],xx[selpn]+DXS,20*np.log10(np.abs(Gpe[selpn])));dPn-=dPn.mean()
dFn=20*np.log10(np.abs(Gf[seln]));dFn-=dFn.mean()
print('pec   near x=12-17 FDTD-PE RMS  = %.3f dB'%np.sqrt(np.mean((dFn-dPn)**2)))
print('screen FDTD-PE    RMS          = %.3f dB ; FDTD-Fresnel = %.3f dB'%(eSc,eScF))
np.savez('fdtd_errors.npz',eFree=eFree,eFreeFD=eFreeFD,ePec=ePec,ePecP=ePecP,ePecFP=ePecFP,eSc=eSc,eScF=eScF)
print('saved fig_fdtd_validation.png')
