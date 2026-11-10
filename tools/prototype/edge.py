import numpy as np, cv2, math

def odd(n): 
    n=int(n); return n if n%2==1 else n+1

def order_points(p):
    p=np.array(p,np.float32)
    s=p.sum(1); d=p[:,1]-p[:,0]
    return np.array([p[s.argmin()],p[d.argmin()],p[s.argmax()],p[d.argmax()]],np.float32)

# ---------------- cues ----------------
def text_mask(gray, strong_fac=0.5, center_win=None, center_fac=0.35):
    """gray uint8 (closed not applied). Returns uint8 mask of text strokes (hysteresis)."""
    h,w=gray.shape
    k=odd(max(9,0.025*max(h,w)))
    bh=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    thr,_=cv2.threshold(bh,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    thr=max(thr,14.0)
    strong=(bh>thr).astype(np.uint8)
    weak=(bh>max(8.0,strong_fac*thr)).astype(np.uint8)
    if center_win is not None:
        x0,x1=center_win
        wk2=(bh>max(6.0,center_fac*thr)).astype(np.uint8)
        weak[:,x0:x1]=wk2[:,x0:x1]
    near=cv2.dilate(strong,np.ones((15,15),np.uint8))
    return (weak&near)

def paper_model(Lc,A,B):
    h,w=Lc.shape
    roi=(slice(int(.3*h),int(.7*h)),slice(int(.3*w),int(.7*w)))
    l=Lc[roi]; P=float(np.percentile(l,85))
    sel=l>=np.percentile(l,60)
    ma=float(np.median(A[roi][sel])); mb=float(np.median(B[roi][sel]))
    return P,ma,mb

def detect_quad(bgr, debug=None):
    H,W=bgr.shape[:2]
    sc=700.0/max(H,W)
    small=cv2.resize(bgr,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
    h,w=small.shape[:2]; long=max(h,w)
    lab=cv2.cvtColor(small,cv2.COLOR_BGR2Lab)
    L=lab[:,:,0].astype(np.float32); A=lab[:,:,1].astype(np.float32)-128; B=lab[:,:,2].astype(np.float32)-128
    kc=odd(max(9,0.02*long))
    Lc=cv2.morphologyEx(L,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kc,kc)))
    Lc=cv2.GaussianBlur(Lc,(0,0),1.0)
    P,ma,mb=paper_model(Lc,A,B)
    bright=np.clip((Lc-0.30*P)/(0.30*P),0,1)
    d=np.hypot(A-ma,B-mb)
    chrom=np.exp(-(d/12.0)**2)
    chrom=cv2.GaussianBlur(chrom.astype(np.float32),(0,0),2.0)
    Pm=(bright*chrom).astype(np.float32)

    # ---- rough region ----
    m=(Pm>0.5).astype(np.uint8)*255
    k=odd(0.02*long)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)))
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(k,k)))
    # nối hai nửa giấy bị bóng gáy chia cắt (đóng theo chiều ngang, kernel rộng)
    kw=odd(0.16*long)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(kw,3)))
    cs,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    best=None;bs=0
    for c in cs:
        a=cv2.contourArea(c)
        if a<0.12*h*w: continue
        inside=cv2.pointPolygonTest(c,(w/2,h/2),False)>=0
        s=a*(1 if inside else .3)
        if s>bs: bs=s;best=c
    R0=np.zeros((h,w),np.uint8)
    if best is not None: cv2.drawContours(R0,[best],-1,255,-1)
    full = best is None or (R0>0).mean()>0.93

    # ---- text ----
    tm=text_mask(np.clip(L,0,255).astype(np.uint8))
    if not full:
        er=cv2.erode(R0,cv2.getStructuringElement(cv2.MORPH_RECT,(odd(.02*long),)*2))
        tm&=(er>0).astype(np.uint8)
    blocks=cv2.dilate(tm,cv2.getStructuringElement(cv2.MORPH_RECT,(odd(.03*w),odd(.015*h))))
    n,lab_cc,stats,_=cv2.connectedComponentsWithStats(blocks)
    keep=np.zeros(n,bool)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=0.004*h*w: keep[i]=True
    blocks=keep[lab_cc]
    ys,xs=np.nonzero(tm&blocks)
    if len(xs)>4000:
        idx=np.random.RandomState(1).choice(len(xs),4000,replace=False); ys,xs=ys[idx],xs[idx]
    tpts=np.stack([xs,ys],1).astype(np.float32) if len(xs)>50 else None

    # ---- rough quad ----
    def rect_quad(pts):
        r=cv2.minAreaRect(pts.astype(np.float32)); return order_points(cv2.boxPoints(r))
    if not full:
        hull=cv2.convexHull(best)
        peri=cv2.arcLength(hull,True); quad=None; eps=.01
        while eps<=.12:
            ap=cv2.approxPolyDP(hull,eps*peri,True)
            if len(ap)==4: quad=ap.reshape(4,2).astype(np.float32);break
            if len(ap)<4:break
            eps+=.005
        rough=order_points(quad if quad is not None else rect_quad(hull.reshape(-1,2)))
    elif tpts is not None:
        q=rect_quad(tpts); c=q.mean(0); rough=order_points(c+(q-c)*1.18)
    else:
        mg=.03; rough=np.array([[mg*w,mg*h],[(1-mg)*w,mg*h],[(1-mg)*w,(1-mg)*h],[mg*w,(1-mg)*h]],np.float32)
    rough_area=cv2.contourArea(rough)
    cen=rough.mean(0)

    Ls=cv2.GaussianBlur(Lc,(0,0),0.8)
    def bil(img,x,y):
        x=np.clip(x,0,w-1.001);y=np.clip(y,0,h-1.001)
        x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);fx=x-x0;fy=y-y0
        return (img[y0,x0]*(1-fx)*(1-fy)+img[y0,x0+1]*fx*(1-fy)+img[y0+1,x0]*(1-fx)*fy+img[y0+1,x0+1]*fx*fy)

    band=int(0.12*long); K=28
    scales=[2,4,8,14]; gam=[1,.97,.93,.88]
    lines=[]; dbg=[]
    for i in range(4):
        a=rough[i]; b=rough[(i+1)%4]
        dvec=(b-a); sl=np.linalg.norm(dvec); dvec/=sl
        o=np.array([dvec[1],-dvec[0]],np.float32)
        if np.dot(o,(a+b)/2-cen)<0: o=-o
        # text extent along o (conservative)
        if tpts is not None:
            proj=tpts@o; tmax=float(np.percentile(proj,99.7))
        else: tmax=-1e9
        marg=0.006*long
        us=np.arange(-band,band+1)
        cand=[]
        for kk in range(K):
            t=0.08+0.84*kk/(K-1)
            c=a+(b-a)*t
            xs_=c[0]+us*o[0]; ys_=c[1]+us*o[1]
            pm=bil(Pm,xs_,ys_); lv=bil(Ls,xs_,ys_)
            inb=(xs_>=0)&(xs_<=w-1)&(ys_>=0)&(ys_<=h-1)
            pos=xs_*o[0]+ys_*o[1]
            cp=np.concatenate([[0],np.cumsum(pm)]); cl=np.concatenate([[0],np.cumsum(lv)])
            n=len(us); Bs=np.zeros(n); Lq=np.zeros(n)
            for s,g in zip(scales,gam):
                for j in range(s,n-s):
                    pin=(cp[j]-cp[j-s])/s; pout=(cp[j+s+1]-cp[j+1])/s
                    Bs[j]=max(Bs[j],g*np.clip(pin-pout,0,1))
                    lin=(cl[j]-cl[j-s])/s; lout=(cl[j+s+1]-cl[j+1])/s
                    dl=(lin-lout)/(0.30*P)
                    Lq[j]=max(Lq[j],g*(np.clip(dl,0,1)+0.5*np.clip(-dl,0,1)))
            far=25; cons=np.zeros(n)
            for j in range(far,n):
                cons[j]=(cp[j]-cp[j-far])/far
            cons=0.4+0.6*np.clip((cons-0.3)/0.5,0,1)
            gate=np.clip((pos-tmax)/max(marg,1),0.05,1.0)
            prior=np.exp(-0.5*((pos-(tmax+0.05*long))/(0.07*long))**2) if tmax>-1e8 else np.zeros(n)
            S=gate*((0.5*Bs+0.35*Lq)*cons+0.12*prior)
            S[~inb]=0
            S[:max(far,15)]=0; S[-15:]=0
            j=int(S.argmax()); smax=S[j]
            ex=S.copy(); ex[max(0,j-8):j+9]=-1; s2=max(ex.max(),float(np.median(S)))
            conf=float(np.clip((smax-s2)/0.35,0,1))
            if 1<j<n-2:
                den=S[j-1]-2*S[j]+S[j+1]
                off=0.5*(S[j-1]-S[j+1])/den if abs(den)>1e-6 else 0
                off=float(np.clip(off,-1,1))
            else: off=0
            u=us[j]+off
            cand.append((c+o*u,conf,kk))
        pts=np.array([x[0] for x in cand]);cf=np.array([x[1] for x in cand])
        # RANSAC weighted line
        tol=0.006*long
        rng=np.random.RandomState(7+i)
        bestline=None;bestsc=0;bestin=None
        valid=np.nonzero(cf>0.15)[0]
        if len(valid)>=4:
            for _ in range(100):
                p,q=rng.choice(valid,2,replace=False)
                dv=pts[q]-pts[p]; ln=np.linalg.norm(dv)
                if ln<0.2*sl: continue
                dv/=ln; nv=np.array([-dv[1],dv[0]])
                dist=np.abs((pts-pts[p])@nv)
                inl=(dist<tol)&(cf>0.15)
                sc_=cf[inl].sum()
                if sc_>bestsc: bestsc=sc_;bestin=inl
        line=None
        if bestin is not None and bestin.sum()>=4:
            wgt=cf[bestin];pp=pts[bestin]
            mu=(pp*wgt[:,None]).sum(0)/wgt.sum()
            cov=((pp-mu).T*wgt)@(pp-mu)
            ev,evec=np.linalg.eigh(cov); dv=evec[:,1]
            line=(mu,dv/np.linalg.norm(dv),True)
        else:
            line=((a+b)/2,dvec,False)
        lines.append((line,o,tmax))
        dbg.append((pts,cf,bestin))
    # ---- contain text: shift each line outward if text pokes out ----
    adj=[]
    for (mu,dv,ok),o,tmax in lines:
        nv=np.array([dv[1],-dv[0]])
        if np.dot(nv,o)<0: nv=-nv
        if tpts is not None:
            v=float(np.max((tpts-mu)@nv))
            need=v+0.006*long
            if need>0: mu=mu+nv*need
        adj.append((mu,dv,ok))
    def inter(l1,l2):
        p1,d1,_=l1;p2,d2,_=l2
        Am=np.array([d1,-d2]).T
        if abs(np.linalg.det(Am))<1e-6: return None
        t=np.linalg.solve(Am,p2-p1)
        return p1+d1*t[0]
    q=[inter(adj[(i-1)%4],adj[i]) for i in range(4)]  # corner i between side i-1 and side i
    okq=all(x is not None for x in q)
    if okq:
        q=np.array(q,np.float32); q=order_points(q)
        area=cv2.contourArea(q)
        convex=cv2.isContourConvex(q.reshape(-1,1,2))
        inside=np.all((q[:,0]>-0.05*w)&(q[:,0]<1.05*w)&(q[:,1]>-0.05*h)&(q[:,1]<1.05*h))
        okq=convex and inside and 0.5*rough_area<area<1.6*rough_area
    if not okq: q=rough
    if debug is not None: debug.update(dict(rough=rough,full=full,Pm=Pm,tm=tm,dbg=dbg,sc=sc))
    q=np.clip(q,[0,0],[w-1,h-1])
    return order_points(q/sc)

# ---------------- gutter ----------------
def find_cut(spread, debug=None):
    H,W=spread.shape[:2]
    sc=700.0/W
    small=cv2.resize(spread,None,fx=sc,fy=sc,interpolation=cv2.INTER_AREA)
    h,w=small.shape[:2]
    g=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY)
    kc=odd(max(9,0.025*max(h,w)))
    Lc=cv2.morphologyEx(g.astype(np.float32),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kc,kc)))
    x0,x1=int(.3*w),int(.7*w)
    tm=text_mask(g,center_win=(x0,x1))
    # ignore top/bottom/side 3% where borders live
    mx,my=int(.03*w),int(.03*h)
    tm[:my]=0;tm[-my:]=0;tm[:,:mx]=0;tm[:,-mx:]=0
    colmean=Lc[int(.1*h):int(.9*h)].mean(0)
    k=odd(max(5,w/40.0))
    cm=cv2.GaussianBlur(colmean.reshape(1,-1),(k,1),0).ravel()
    ref=float(np.percentile(cm[int(.1*w):int(.9*w)],85))
    D=np.clip(1-cm/ref,0,1); Dn=np.clip(D/0.15,0,1)
    dens=tm.sum(0)/float(h)
    tc=cv2.dilate((dens>0.004).astype(np.uint8).reshape(1,-1),np.ones((1,3),np.uint8)).ravel()>0
    dist=np.zeros(w,np.float32); last=-1e9
    for x in range(w):
        if tc[x]: last=x
        dist[x]=x-last
    last=1e9
    for x in range(w-1,-1,-1):
        if tc[x]: last=x
        dist[x]=min(dist[x],last-x)
    gap=np.clip(dist/(0.025*w),0,1)
    xs=np.arange(w)
    prior=np.exp(-0.5*((xs-w/2)/(0.08*w))**2)
    S=0.40*Dn+0.40*gap+0.20*prior
    S[:x0]=-1;S[x1:]=-1
    xb=int(S.argmax())
    m=int(max(2,0.008*w))
    xl=xb
    while xl>0 and not tc[xl]: xl-=1
    xr=xb
    while xr<w-1 and not tc[xr]: xr+=1
    risky=False
    if xr-xl>=2*m+2:
        xf=int(np.clip(xb,xl+m,xr-m))
    else:
        xf=xb; risky=True
    if debug is not None: debug.update(dict(S=S,Dn=Dn,gap=gap,xl=xl,xr=xr,risky=risky,tm=tm))
    return int(round(xf/sc)),risky
