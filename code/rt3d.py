"""
Coherent 3D ray tracing (GO + Fresnel edge diffraction) for the urban block.
For y=0 receivers the rooftop problem is exactly the validated 2D x-z EP
(cylindrical z spreading, kuttler_2d.py); the third transverse dimension adds
a spherical Gaussian spreading factor in y. Finite-width vertical side edges
add 3D conical diffraction. Complex fields, phase retained.
"""
import numpy as np
from scipy.special import fresnel
f=900e6;c=3e8;lam=c/f;k0=2*np.pi/lam
eps_c=5.0-1j*0.01/(2*np.pi*f*8.854e-12)
def sc(xc,t=3.):return (xc-t/2,xc+t/2)
blds=[(*sc(205),-30,30,10),(*sc(375),-30,5,25),(*sc(545),-5,30,16),(*sc(705),-30,30,12),(*sc(875),-20,30,30)]
Tx=(0.,0.,20.);sig=3.;xR=k0*sig**2
def fR(g):
    n=np.sqrt(eps_c);g=np.clip(g,1e-4,np.pi/2-1e-4)
    return (np.sin(g)-np.sqrt(n**2-np.cos(g)**2))/(np.sin(g)+np.sqrt(n**2-np.cos(g)**2))
def kF(v):
    C,S=fresnel(v);return (1+1j)/2*((0.5-C)+1j*(S-0.5))
def sxs(s):return sig*np.sqrt(1+(s/xR)**2)
def gb(s,dz):return np.sqrt(sig/sxs(s))*np.exp(-dz**2/(2*sxs(s)**2))   # cylindrical (one dim)
def gby(s,dy):return np.sqrt(sig/sxs(s))*np.exp(-dy**2/(2*sxs(s)**2))  # y spherical factor
def lh(x0,z0,x1,z1,x):return z0+(z1-z0)*(x-x0)/(x1-x0+1e-12)
def screens_at(yr):
    """Rooftop edges whose finite y-span covers the Fermat y coordinate at y=0 projection."""
    out=[]
    for x0,x1,y1,y2,h in blds:
        out.append((x0,h,y1,y2))
    return out
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
    """2D x-z EP (direct+cascaded rooftop+ground reflection), cylindrical."""
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
    # receiver inside a building block: zero field
    for x0,x1,y1,y2,h in blds:
        if x0<=xr<=x1 and y1<=yr<=y2 and zr<=h:
            return 0+0j
    # y=0 vertical plane: rooftop edges that cover the Fermat y projection
    edges0=[(x0,h,y1,y2) for x0,x1,y1,y2,h in blds if y1-1e-9<=yr<=y2+1e-9]
    E=rt2d_profile(xr,zr,ground,edges0)*gby(xr,yr-Tx[1])
    # upper corner (rooftop x side-edge) diffraction: two Fresnel factors
    ylos=Tx[1]+(yr-Tx[1])*np.array([x0 for x0,x1,y1,y2,h in blds])/xr
    zlos=Tx[2]+(zr-Tx[2])*np.array([x0 for x0,x1,y1,y2,h in blds])/xr
    for (x0,x1,y1,y2,h),yl,zl in zip(blds,ylos,zlos):
        if not 0<x0<xr:continue
        for ye in (y1,y2):
            if zl>h:continue  # ray passes above rooftop, corner not illuminated
            d1=np.hypot(np.hypot(x0,ye-Tx[1]),h-Tx[2])
            d2=np.hypot(np.hypot(xr-x0,yr-ye),zr-h)
            if d2<=0:continue
            vy=abs(ye-yl)*np.sqrt(2/lam*(1/d1+1/d2))
            vz=max(h-zl,0)*np.sqrt(2/lam*(1/d1+1/d2))
            F=kF(vy)*kF(vz)
            E+=(gby(d1,ye-Tx[1])*gb(d1,h-Tx[2])*gby(d2,yr-ye)*gb(d2,zr-h)
                *F*np.exp(-1j*k0*(d1+d2)))
    return E
if __name__=='__main__':
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    xs=np.linspace(20,990,400);zr=15.
    Ec=np.array([rt3d(x,0,zr,'conc') for x in xs]);Ep=np.array([rt3d(x,0,zr,'pec') for x in xs])
    d=np.load('urban3d_fields.npz');Hc=d['Hc'];Hp=d['Hp'];xx=d['xx'];y=d['y'];z=d['z'];Txz=float(d['Txz'])
    iy0=np.argmin(np.abs(y));iz=np.argmin(np.abs(z-zr));n0=np.abs(Hc[0,iy0,np.argmin(np.abs(z-Txz))])
    Lc=20*np.log10(n0/(np.abs(Hc[:,iy0,iz])+1e-10));Lp=20*np.log10(n0/(np.abs(Hp[:,iy0,iz])+1e-10))
    plt.figure(figsize=(11,5.5))
    plt.plot(xx,Lc,'b-',lw=2,label='3D PE concrete');plt.plot(xx,Lp,'r-',lw=1.4,label='3D PE PEC')
    plt.plot(xs,-20*np.log10(np.abs(Ec)+1e-12),color='cyan',ls='--',lw=1.6,label='3D ray concrete')
    plt.plot(xs,-20*np.log10(np.abs(Ep)+1e-12),color='magenta',ls='--',lw=1.6,label='3D ray PEC')
    for x0,x1,y1,y2,h in blds:
        if y1<=0<=y2 and h>=zr:plt.axvspan(x0,x1,color='gray',alpha=0.2)
    plt.ylim(0,70);plt.xlabel('Range x (m)');plt.ylabel('Path loss (dB)');plt.legend();plt.grid(alpha=0.3)
    plt.title('3D urban: Kuttler PE vs coherent 3D ray (y=0,z=15)')
    plt.tight_layout();plt.savefig('urban3d_vs_ray.png',dpi=160);plt.close()
    print('saved')
