"""
Multi-receiver validation statistics for the 3D urban block.
PE (stored full 3D field) vs coherent 3D ray tracing over a grid of receiver
lines: heights z = 10,15,20,25 m and cross positions y = -10,0,10 m.
For each line report mean absolute deviation, RMS and 95th percentile, and
decompose the discrepancy by physical regime (LOS / rooftop shadow / deep
shadow behind the tall last building) to attribute error sources.
"""
import numpy as np
from rt3d import rt3d,blds

d=np.load('urban3d_fields.npz');Hc=d['Hc'];y=d['y'];z=d['z'];X=d['xx'];Txz=float(d['Txz'])
iy0=np.argmin(np.abs(y));iz0=np.argmin(np.abs(z-Txz));n0=np.abs(Hc[0,iy0,iz0])

def inside(x,yr,zr):
    for x0,x1,y1,y2,h in blds:
        if x0<=x<=x1 and y1<=yr<=y2 and zr<=h: return True
    return False

zs=[10,15,20,25]; ys=[-10,0,10]
xs=np.linspace(20,995,200)
rows=[]
print('z(m) y(m) |  mean   RMS   p95  | LOS  roof  deep')
for zr in zs:
    for yr in ys:
        iy=np.argmin(np.abs(y-yr));iz=np.argmin(np.abs(z-zr))
        Lpe=np.array([20*np.log10(n0/(np.abs(Hc[np.argmin(np.abs(X-x)),iy,iz])+1e-12)) for x in xs])
        Er=np.array([rt3d(x,yr,zr,'conc') for x in xs],dtype=complex)
        Lr=-20*np.log10(np.abs(Er)+1e-12)
        msk=np.array([not inside(x,yr,zr) and abs(E)>1e-9 for x,E in zip(xs,Er)])
        msk &= xs>30
        err=np.abs(Lpe-Lr)[msk]
        # regimes
        los=msk & (xs<200)
        roof=msk & (xs>=200)&(xs<870)
        deep=msk & (xs>=875)
        def stat(m):
            e=np.abs(Lpe-Lr)[m]; return np.sqrt((e**2).mean()) if e.size else np.nan
        rows.append((zr,yr,err.mean(),np.sqrt((err**2).mean()),np.percentile(err,95),
                     stat(los),stat(roof),stat(deep),msk.sum()))
        print('%4d %4d | %5.2f %5.2f %5.2f | %4.1f %5.2f %5.2f'%(
            zr,yr,rows[-1][2],rows[-1][3],rows[-1][4],rows[-1][5],rows[-1][6],rows[-1][7]))

R=np.array(rows)
allmean=np.nanmean(R[:,2]);allrms=np.sqrt(np.nanmean(R[:,3]**2));allp95=np.nanmean(R[:,4])
print('\nPOOLED across %d lines: mean=%.2f RMS=%.2f p95=%.2f dB'%(len(rows),allmean,allrms,allp95))
print('regime RMS (pooled): LOS=%.2f rooftop=%.2f deep=%.2f'%(
    np.nanmean(R[:,5]),np.nanmean(R[:,6]),np.nanmean(R[:,7])))
np.savez('urban_multiline.npz',R=R,zs=zs,ys=ys)
