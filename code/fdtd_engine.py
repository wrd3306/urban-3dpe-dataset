"""
Independent 2D full-wave FDTD reference (TM-Ez) with CPML, used to validate
the SSFT-PE in three small canonical problems WITHOUT relying on ray tracing:
  (1) free-space cylindrical spreading from a point source,
  (2) PEC ground image interference,
  (3) diffraction behind a PEC half-plane knife screen.
TM-Ez: Ez is transverse to the propagation (x-z) plane; on a PEC surface the
tangential E (Ez) vanishes, i.e. the Dirichlet (soft) screen the 2D PE uses.
Continuous-wave 900 MHz excitation; complex amplitude by I/Q demodulation.
Run each scenario separately (argv: free, pec, screen).
"""
import sys, time
import numpy as np

C0=3e8; EPS0=8.8541878128e-12; MU0=1.25663706212e-6; ETA0=376.7303
f0=900e6; lam=C0/f0; k0=2*np.pi/lam; omega=2*np.pi*f0
DD=0.025                 # 0.025 m = lambda/13.3
NPML=12

def cpml_coeff(n,delta):
    """1D CNR-CPML b,c coefficients along one axis (interior b=1,c=0).
    Normalised depth rho grows from 0 at the inner edge (facing the
    interior, sigma->0 for impedance matching) to 1 at the outer boundary
    (sigma maximum, alpha zero)."""
    b=np.ones(n);c=np.zeros(n);m=3.0
    sigmax=0.8*(m+1)/(ETA0*delta); almax=0.05
    dt=0.99*delta/(C0*np.sqrt(2))
    for i in range(NPML):
        rho=(NPML-i)/NPML        # 1 at outer wall (i=0), ~0 at inner edge
        sig=sigmax*rho**m
        al=almax*(1.0-rho)
        for edge in (i,n-1-i):
            b[edge]=np.exp(-(sig+al)*dt/EPS0)
            c[edge]=(sig/(sig+al))*(b[edge]-1.0)
    return b,c

def simulate(scen='free',Dx=40.,Dz=24.,xsm=5.,zsm=12.,xe=20.,h=6.,
             nsteps=12000,verbose=True):
    Nx=int(Dx/DD);Nz=int(Dz/DD)
    ixs=int(xsm/DD);jzs=int(zsm/DD)
    Ez=np.zeros((Nx,Nz));Hx=np.zeros((Nx,Nz));Hz=np.zeros((Nx,Nz))
    pz=np.zeros((Nx,Nz));qx=np.zeros((Nx,Nz));px=np.zeros((Nx,Nz));qz=np.zeros((Nx,Nz))
    ground=scen=='pec'; screen=scen=='screen'
    bx,cx=cpml_coeff(Nx,DD);bz,cz=cpml_coeff(Nz,DD)
    if ground:
        # The bottom is a PEC (Dirichlet) reflector: it must reflect, so the
        # bottom CPML is disabled (PEC z=0 is the physical boundary). Only the
        # left/right/top edges absorb.
        bz[0:NPML]=1.0;cz[0:NPML]=0.0
    bx2=bx[:,None];cx2=cx[:,None];bz2=bz[None,:];cz2=cz[None,:]
    dt=0.99*DD/(C0*np.sqrt(2)); CH=dt/MU0;CE=dt/EPS0
    jh=int(h/DD);ixe=int(xe/DD)
    per=int(round((1/f0)/dt));tau=6/f0
    sR=np.zeros((Nx,Nz));sI=np.zeros((Nx,Nz));samp_start=nsteps-2*per
    t0=time.time()
    for n in range(nsteps):
        t=n*dt
        # ---- H update (uses previous-step memory) ----
        dEz=(Ez[:,1:Nz]-Ez[:,0:Nz-1])/DD               # dEz/dz at (i,j-1/2)
        Hx[:,1:Nz]=Hx[:,1:Nz]-CH*(dEz+pz[:,1:Nz])
        dEx=(Ez[1:Nx,:]-Ez[0:Nx-1,:])/DD               # dEz/dx at (i-1/2,j)
        Hz[1:Nx,:]=Hz[1:Nx,:]+CH*(dEx+qx[1:Nx,:])
        # ---- CPML memory for H ----
        pz[:,1:Nz]=bz2[:,1:Nz]*pz[:,1:Nz]+cz2[:,1:Nz]*dEz
        qx[1:Nx,:]=bx2[1:Nx,:]*qx[1:Nx,:]+cx2[1:Nx,:]*dEx
        # ---- Ez interior update: curl H at (i,j), i=1..Nx-2,j=1..Nz-2 ----
        dHzx=(Hz[2:Nx,1:Nz-1]-Hz[1:Nx-1,1:Nz-1])/DD    # dHz/dx at (i,j)
        dHxz=(Hx[1:Nx-1,2:Nz]-Hx[1:Nx-1,1:Nz-1])/DD    # dHx/dz at (i,j)
        px[1:Nx-1,1:Nz-1]=bx2[1:Nx-1,:]*px[1:Nx-1,1:Nz-1]+cx2[1:Nx-1,:]*dHzx
        qz[1:Nx-1,1:Nz-1]=bz2[:,1:Nz-1]*qz[1:Nx-1,1:Nz-1]+cz2[:,1:Nz-1]*dHxz
        Ez[1:Nx-1,1:Nz-1]+=CE*(dHzx+px[1:Nx-1,1:Nz-1]-dHxz-qz[1:Nx-1,1:Nz-1])
        # ---- PEC ----
        if ground: Ez[:,0]=0.0
        if screen:
            Ez[ixe,0:jh+1]=0.0; Hx[ixe,0:jh+1]=0.0
        # ---- soft CW source with Hann ramp ----
        ramp=1.0 if t>tau else 1.0-np.exp(-(t/tau)**2)
        Ez[ixs,jzs]+=np.sin(omega*t)*ramp*dt/EPS0
        # ---- I/Q sampling (last 2 periods) ----
        if n>=samp_start:
            w=2.0/per
            sR+=w*Ez*np.cos(omega*t); sI-=w*Ez*np.sin(omega*t)
        if verbose and (n+1)%2000==0:
            print('  %s step %d/%d (%.0f s)'%(scen,n+1,nsteps,time.time()-t0))
    Ec=sR+1j*sI
    return np.arange(Nx)*DD,np.arange(Nz)*DD,Ec,(ixs,jzs)

if __name__=='__main__':
    scen=sys.argv[1] if len(sys.argv)>1 else 'free'
    nsteps=int(sys.argv[2]) if len(sys.argv)>2 else 6000
    if len(sys.argv)>3:
        DD=float(sys.argv[3])              # refined grid size (m)
    suffix=sys.argv[4] if len(sys.argv)>4 else ''
    if scen=='free':   x,z,Ec,src=simulate('free',25,20,4,10,nsteps=nsteps)
    elif scen=='pec':  x,z,Ec,src=simulate('pec',25,20,4,8,nsteps=nsteps)
    else:              x,z,Ec,src=simulate('screen',30,18,4,10,xe=12,h=8,nsteps=nsteps)
    np.savez('fdtd_%s%s.npz'%(scen,suffix),x=x,z=z,Ec=Ec,src=np.array(src),
             lam=lam,k0=k0,DD=DD,nsteps=nsteps)
    print('saved fdtd_%s%s.npz'%(scen,suffix))
