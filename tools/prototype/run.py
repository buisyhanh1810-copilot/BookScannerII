import numpy as np, cv2, sys, time
from synth import *
import edge

def evaluate(img,gt,method):
    t=time.time()
    q=edge.detect_quad(img) if method=='new' else old_quad(img)
    dt=time.time()-t
    if q is None: return None
    Hc,Wc=img.shape[:2]; diag=np.hypot(Wc,Hc)
    qg=order_points(gt['quad'])
    cerr=np.linalg.norm(q-qg,axis=1).max()/diag*100   # % of image diagonal
    # text containment: warp flat text masks with detected quad
    spread,M=warp(img,q)
    wl=cv2.warpPerspective(gt['wl'],M,(spread.shape[1],spread.shape[0]))
    wr=cv2.warpPerspective(gt['wr'],M,(spread.shape[1],spread.shape[0]))
    full_l=(gt['wl']>0).sum(); full_r=(gt['wr']>0).sum()
    # fraction of text lost (outside crop)
    inside=cv2.warpPerspective(np.full(gt['wl'].shape,255,np.uint8),M,(spread.shape[1],spread.shape[0]))
    # lost text: transform original text mask pixel counts inside warped area
    ws=cv2.warpPerspective(gt['wl']|gt['wr'],M,(spread.shape[1],spread.shape[0]),flags=cv2.INTER_NEAREST)
    lost=1-(ws>0).sum()/max(1,((gt['wl']|gt['wr'])>0).sum()*(M[0,0]*M[1,1]-M[0,1]*M[1,0]))
    # gutter
    if method=='new': xc,risky=edge.find_cut(spread)
    else: xc,risky=old_gutter(spread),False
    leftpage=(wr[:,:xc]>0).sum(); rightpage=(wl[:,xc:]>0).sum()   # leakage of other page's text
    cutlen=(( (wl|wr)[:,max(0,xc-1):xc+2])>0).sum()
    tot=((wl|wr)>0).sum()
    return dict(cerr=cerr,leak=(leftpage+rightpage)/max(1,tot)*100,cut=cutlen,lostfrac=None,dt=dt,xc=xc,W=spread.shape[1])

cases=[]
seed=1
for kind in ['dark','white','wood','gray']:
    for rot,tilt in [(0,0),(6,1),(-12,2),(20,1.5)]:
        cases.append((seed,kind,rot,tilt)); seed+=1
res={'old':[],'new':[]}
for sd,kind,rot,tilt in cases:
    img,gt=make_case(sd,kind,rot,tilt)
    line=f"{kind:5s} rot={rot:4d} tilt={tilt:3.1f} |"
    for m in ['old','new']:
        r=evaluate(img,gt,m)
        if r is None: line+=f" {m}: FAIL |"; res[m].append(None); continue
        res[m].append(r)
        line+=f" {m}: cornerErr={r['cerr']:5.2f}% leak={r['leak']:5.2f}% cutTextPx={r['cut']:4d} |"
    print(line)
for m in ['old','new']:
    ok=[r for r in res[m] if r]
    print(m,'fail',len(res[m])-len(ok),'meanCornerErr %.2f'%np.mean([r['cerr'] for r in ok]),'mean leak %.2f'%np.mean([r['leak'] for r in ok]),'worst corner %.2f'%max(r['cerr'] for r in ok))
