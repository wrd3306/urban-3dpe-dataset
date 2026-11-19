"""
M8 parameter-sensitivity extension (real 3D SSFT-PE + coherent 3D ray).
Sweeps the three reviewer-suggested axes, keeping the production grid fixed:
  (1) carrier frequency  f = 700 / 900 / 1800 MHz  (source 20 m, staggered layout)
  (2) source height      Tx_z = 10 / 20 / 30 m     (900 MHz, staggered layout)
  (3) building layout    staggered baseline vs regular uniform block (900 MHz, 20 m)
For every case the full 3D PE is marched and compared with the coherent 3D ray
model over the SAME twelve receiver lines (z=10,15,20,25 m; y=-10,0,10 m),
reporting pooled mean/RMS/95th percentile and the LOS/rooftop/deep-shadow RMS.
The physical grid is held fixed (Ly=120/Ny=192, H=60/Nz=4096, dx=1.25) so that
the sweep reflects a fixed deployed grid; the lateral Nyquist angle is reported
per frequency. Concrete epsilon follows the same sigma=0.01 S/m model at every f.
"""
import time
import numpy as np
from scipy.special import fresnel
from kuttler_core3d import Kuttler3D

c=3e8; eps0=8.854e-12; sigma=0.01
Ly=120.;Ny=192;H=60.;Nz=4096;abs_frac=0.16
Lx=1000.;dx=1.25;Nx=int(Lx/dx); X=np.arange(Nx)*dx
sig=3.

def sc(xc,t=3.):return (xc-t/2,xc+t/2)
BLD_STAG=[(*sc(205),-30,30,10),(*sc(375),-30,5,25),(*sc(545),-5,30,16),
          (*sc(705),-30,30,12),(*sc(875),-20,30,30)]
BLD_REG=[(*sc(xc),-30,30,18.) for xc in (150,320,490,660,830)]

# ---------------- 3D PE march (parameterised copy of urban_lib.run_urban) ----
def march(f,txz,blds,pol='H'):
    lam=c/f;k0=2*np.pi/lam
    eps_c=5.0-1j*sigma/(2*np.pi*f*eps0)
    S=Kuttler3D(f,Ly=Ly,Ny=Ny,H=H,Nz=Nz,abs_frac=abs_frac)
    S.set_ground(eps_c=eps_c,pol=pol)
    y=S.y;z=S.z
    Y,Z=np.meshgrid(y,z,indexing='ij')
    u=np.exp(-(Y-0.)**2/(2*sig**2))*np.exp(-(Z-txz)**2/(2*sig**2))
    hist=np.zeros((Nx,S.Ny,S.Nz),dtype=complex)
    for i in range(Nx):
        hist[i]=u
        xv=i*dx
        for x0,x1,y1,y2,h in blds:
            if x0<=xv<x0+dx:
                u[(Y>=y1)&(Y<=y2)&(Z<=h)]=0
        u=S.step(u,dx)
    return hist,y,z,lam,k0,eps_c

# ---------------- coherent 3D ray (parameterised copy of rt3d) ---------------
def make_ray(f,eps_c,blds,txy,txz):
    lam=c/f;k0=2*np.pi/lam;n=np.sqrt(eps_c);xR=k0*sig**2
    Tx=(0.,txy,txz)
    def fR(g):
        g=np.clip(g,1e-4,np.pi/2-1e-4)
        return (np.sin(g)-np.sqrt(n**2-np.cos(g)**2))/(np.sin(g)+np.sqrt(n**2-np.cos(g)**2))
    def kF(v):
        C,S=fresnel(v);return (1+1j)/2*((0.5-C)+1j*(S-0.5))
    def sxs(s):return sig*np.sqrt(1+(s/xR)**2)
    def gb(s,dz):return np.sqrt(sig/sxs(s))*np.exp(-dz**2/(2*sxs(s)**2))
    def gby(s,dy):return np.sqrt(sig/sxs(s))*np.exp(-dy**2/(2*sxs(s)**2))
    def lh(x0,z0,x1,z1,x):return z0+(z1-z0)*(x-x0)/(x1-x0+1e-12)
    def ep_range(x0,z0,x1,z1,edges):
        active=[(xe,h) for xe,h,y1,y2 in edges if x0<xe<x1 and h>lh(x0,z0,x1,z1,xe)+1e-9]
        nodes=[(x0,z0)]+active+[(x1,z1)];F=1+0j;L=0
        for i in range(1,len(nodes)-1):
            a0,b0=nodes[i-1];a1,b1=nodes[i];a2,b2=nodes[i+1]
            d1=np.hypot(a1-a0,b1-b0);d2=np.hypot(a2-a1,b2-b1)
            F*=kF((b1-lh(a0,b0,a2,b2,a1))*np.sqrt(2/lam*(1/d1+1/d2)));L+=d1
        L+=np.hypot(nodes[-1][0]-nodes[-2][0],nodes[-1][1]-nodes[-2][1]);return F,L
    def seg_blocked(x0,z0,x1,z1,edges):
        for xe,h,y1,y2 in edges:
            if min(x0,x1)<xe<max(x1,x0) and h>lh(x0,z0,x1,z1,xe)+1e-9:return True
        return False
    def rt2d_profile(rx,rz,ground,edges):
        Fd,dd=ep_range(0,Tx[2],rx,rz,edges);E=gb(dd,rz-Tx[2])*Fd*np.exp(-1j*k0*dd)
        if Tx[2]+rz>0:
            px=rx*Tx[2]/(Tx[2]+rz)
            if 0<=px<=rx and not seg_blocked(0,Tx[2],px,0,edges):
                dr=np.hypot(rx,rz+Tx[2]);g=np.arcsin(max(rz,Tx[2])/dr)
                Gamma=-1 if ground=='pec' else fR(g)
                sc2=[(xe,h,y1,y2) for xe,h,y1,y2 in edges if xe>px]
                Fr,_=ep_range(px,0,rx,rz,sc2)
                E+=gb(dr,rz+Tx[2])*Gamma*Fr*np.exp(-1j*k0*dr)
        return E
    def rt3d(xr,yr,zr,ground='conc'):
        for x0,x1,y1,y2,h in blds:
            if x0<=xr<=x1 and y1<=yr<=y2 and zr<=h:return 0+0j
        edges0=[(x0,h,y1,y2) for x0,x1,y1,y2,h in blds if y1-1e-9<=yr<=y2+1e-9]
        E=rt2d_profile(xr,zr,ground,edges0)*gby(xr,yr-Tx[1])
        ylos=Tx[1]+(yr-Tx[1])*np.array([x0 for x0,x1,y1,y2,h in blds])/xr
        zlos=Tx[2]+(zr-Tx[2])*np.array([x0 for x0,x1,y1,y2,h in blds])/xr
        for (x0,x1,y1,y2,h),yl,zl in zip(blds,ylos,zlos):
            if not 0<x0<xr:continue
            for ye in (y1,y2):
                if zl>h:continue
                d1=np.hypot(np.hypot(x0,ye-Tx[1]),h-Tx[2])
                d2=np.hypot(np.hypot(xr-x0,yr-ye),zr-h)
                if d2<=0:continue
                vy=abs(ye-yl)*np.sqrt(2/lam*(1/d1+1/d2))
                vz=max(h-zl,0)*np.sqrt(2/lam*(1/d1+1/d2))
                F=kF(vy)*kF(vz)
                E+=(gby(d1,ye-Tx[1])*gb(d1,h-Tx[2])*gby(d2,yr-ye)*gb(d2,zr-h)
                    *F*np.exp(-1j*k0*(d1+d2)))
        return E
    return rt3d

def inside(blds,x,yr,zr):
    for x0,x1,y1,y2,h in blds:
        if x0<=x<=x1 and y1<=yr<=y2 and zr<=h:return True
    return False

def twelveline(hist,y,z,blds,rt3d,txz):
    iy0=np.argmin(np.abs(y));izt=np.argmin(np.abs(z-txz));n0=np.abs(hist[0,iy0,izt])
    xs=np.linspace(20,995,200);zs=[10,15,20,25];ys=[-10,0,10];rows=[]
    for zr in zs:
        for yr in ys:
            iy=np.argmin(np.abs(y-yr));iz=np.argmin(np.abs(z-zr))
            Lpe=np.array([20*np.log10(n0/(np.abs(hist[np.argmin(np.abs(X-x)),iy,iz])+1e-12)) for x in xs])
            Er=np.array([rt3d(x,yr,zr,'conc') for x in xs],complex)
            Lr=-20*np.log10(np.abs(Er)+1e-12)
            m=np.array([x>30 and not inside(blds,x,yr,zr) and abs(E)>1e-9 for x,E in zip(xs,Er)])
            e=np.abs(Lpe-Lr)
            def st(mm):
                ee=e[mm];return np.sqrt((ee**2).mean()) if ee.size else np.nan
            los=m&(xs<200);roof=m&(xs>=200)&(xs<870);deep=m&(xs>=875)
            rows.append((e[m].mean(),np.sqrt((e[m]**2).mean()),np.percentile(e[m],95),
                         st(los),st(roof),st(deep)))
    R=np.array(rows)
    return dict(mean=np.nanmean(R[:,0]),rms=np.sqrt(np.nanmean(R[:,1]**2)),
                p95=np.nanmean(R[:,2]),los=np.nanmean(R[:,3]),
                roof=np.nanmean(R[:,4]),deep=np.nanmean(R[:,5]))

def run_case(tag,f,txz,blds):
    t0=time.time()
    hist,y,z,lam,k0,eps_c=march(f,txz,blds)
    rt=make_ray(f,eps_c,blds,0.,txz)
    s=twelveline(hist,y,z,blds,rt,txz)
    theta_ny=np.degrees(np.arcsin(lam/(2*(Ly/Ny))))
    print('%-22s f=%5.0fMHz Txz=%2.0d layout=%s | mean=%.2f RMS=%.2f p95=%.2f | LOS=%.2f roof=%.2f deep=%.2f | Nyq=%.1fdeg (%.0fs)'%(
        tag,f/1e6,txz,'reg' if blds is BLD_REG else 'stag',
        s['mean'],s['rms'],s['p95'],s['los'],s['roof'],s['deep'],theta_ny,time.time()-t0),flush=True)
    return s

if __name__=='__main__':
    out={}
    for f in (700e6,900e6,1800e6):
        out['freq_%d'%round(f/1e6)]=run_case('frequency',f,20.,BLD_STAG)
    for txz in (10.,20.,30.):
        out['txz_%d'%txz]=run_case('source height',900e6,txz,BLD_STAG)
    out['layout_regular']=run_case('regular layout',900e6,20.,BLD_REG)
    np.savez('urban_param_sweep.npz',**{k:np.array(list(v.values())) for k,v in out.items()},
             keys=np.array(list(out['freq_900'].keys())))
    print('saved urban_param_sweep.npz')
