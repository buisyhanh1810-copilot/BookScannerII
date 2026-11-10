import numpy as np, cv2, random

WORDS="the of and to in is that for it as was with be by on not he this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no".split()

def flat_spread(seed, FW=1400, FH=1000, text_in=None):
    rng=random.Random(seed); nr=np.random.RandomState(seed)
    img=np.full((FH,FW,3),(228,232,236),np.uint8)  # BGR warm paper
    tm=np.zeros((FH,FW),np.uint8)
    gx=FW//2
    cols=[(95,gx-85),(gx+85,FW-95)]   # outer margin 95, inner margin 85
    for (xa,xb) in cols:
        y=95
        while y<FH-100:
            x=xa
            while x<xb-40:
                wd=rng.choice(WORDS)
                (tw,th),_=cv2.getTextSize(wd,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
                if x+tw>xb: break
                cv2.putText(img,wd,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(40,40,40),2,cv2.LINE_AA)
                cv2.putText(tm,wd,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.7,255,2)
                x+=tw+14
            y+=34
    # side tag: left text mask / right text mask
    # gutter shadow + blur near gutter
    X=np.arange(FW)[None,:,None].astype(np.float32)
    shadow=1-0.55*np.exp(-((X-gx)/55.0)**2)-0.10*np.exp(-((X-gx)/200.0)**2)
    blur=cv2.GaussianBlur(img,(0,0),3.5)
    wmix=np.exp(-((X-gx)/130.0)**2)
    img=(img*(1-wmix)+blur*wmix)
    img=img*shadow
    # outer edge darkening
    edge=np.minimum(X,FW-X); img=img*(1-0.15*np.exp(-edge/25.0))
    # page stack lines on left/right edges
    return np.clip(img,0,255).astype(np.uint8),tm,gx

def background(kind,Wc,Hc,seed):
    nr=np.random.RandomState(seed)
    if kind=='dark': bg=np.full((Hc,Wc,3),(35,38,42),np.uint8)
    elif kind=='white': bg=np.full((Hc,Wc,3),(238,240,242),np.uint8)
    elif kind=='wood':
        bg=np.zeros((Hc,Wc,3),np.uint8); bg[:]=(60,100,150)
        n=nr.randn(1,Wc).astype(np.float32); n=cv2.GaussianBlur(n,(0,0),6)*80
        bg=np.clip(bg+n[0][None,:,None]*np.ones((Hc,1,1)),0,255).astype(np.uint8)
    elif kind=='gray':
        bg=np.full((Hc,Wc,3),(150,150,152),np.uint8)
    bg=np.clip(bg.astype(np.int16)+nr.randn(Hc,Wc,1)*4,0,255).astype(np.uint8)
    return bg

def make_case(seed,kind,rot_deg,tilt,edge_blur=1.6,Wc=2000,Hc=1500,part_blur=True):
    img,tm,gx=flat_spread(seed)
    FH,FW=img.shape[:2]
    rs=np.random.RandomState(seed+100)
    src=np.float32([[0,0],[FW,0],[FW,FH],[0,FH]])
    cx,cy=Wc/2,Hc/2; sz=0.80
    bw,bh_=FW*sz*Wc/FW*0.78/ (Wc/FW)/ (1) ,0
    # destination quad: scale then rotate then perspective jitter
    base=np.float32([[-0.5,-0.5],[0.5,-0.5],[0.5,0.5],[-0.5,0.5]])*np.array([1180,840])
    th=np.deg2rad(rot_deg); R=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]])
    q=base@R.T
    # tilt: perspective-ish shrink of top edge, plus curvature-free
    q[0]+=[tilt*60,tilt*40]; q[1]+=[-tilt*60,tilt*40]
    q[3]+=[-tilt*15,-tilt*10]; q[2]+=[tilt*15,-tilt*10]
    q+=np.float32([cx+rs.randn()*20,cy+rs.randn()*20])
    Hm=cv2.getPerspectiveTransform(src,q.astype(np.float32))
    bg=background(kind,Wc,Hc,seed)
    warped=cv2.warpPerspective(img,Hm,(Wc,Hc),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
    mask=cv2.warpPerspective(np.full((FH,FW),255,np.uint8),Hm,(Wc,Hc))
    mask=cv2.GaussianBlur(mask,(0,0),edge_blur).astype(np.float32)[...,None]/255
    out=(warped*mask+bg*(1-mask))
    # soft cast shadow outside book edges
    out=np.clip(out+np.random.RandomState(seed).randn(Hc,Wc,1)*3,0,255).astype(np.uint8)
    out=cv2.GaussianBlur(out,(0,0),0.8)
    # text masks warped, split by side
    tL=tm.copy(); tL[:,gx:]=0; tR=tm.copy(); tR[:,:gx]=0
    wl=cv2.warpPerspective(tL,Hm,(Wc,Hc)); wr=cv2.warpPerspective(tR,Hm,(Wc,Hc))
    return out,dict(H=Hm,quad=q.astype(np.float32),wl=wl,wr=wr,tl=tL,tr=tR,gx=gx,FW=FW,FH=FH)

def order_points(p):
    p=np.array(p,np.float32); s=p.sum(1); d=p[:,1]-p[:,0]
    return np.array([p[s.argmin()],p[d.argmin()],p[s.argmax()],p[d.argmax()]],np.float32)

def warp(bgr,q):
    tl,tr,br,bl=q
    w=int(max(np.hypot(*(tr-tl)),np.hypot(*(br-bl)))); h=int(max(np.hypot(*(bl-tl)),np.hypot(*(br-tr))))
    dst=np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    M=cv2.getPerspectiveTransform(q.astype(np.float32),dst)
    return cv2.warpPerspective(bgr,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE),M

def old_quad(bgr):
    scale=800.0/max(bgr.shape[:2])
    small=cv2.resize(bgr,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    g=cv2.GaussianBlur(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),(7,7),0)
    _,b=cv2.threshold(g,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    k=cv2.getStructuringElement(cv2.MORPH_RECT,(15,15))
    b=cv2.morphologyEx(b,cv2.MORPH_CLOSE,k);b=cv2.morphologyEx(b,cv2.MORPH_OPEN,k)
    cs,_=cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    ia=small.shape[0]*small.shape[1];c0=(small.shape[1]/2,small.shape[0]/2)
    best=None;bs=0
    for c in cs:
        a=cv2.contourArea(c)
        if a<0.2*ia: continue
        ins=cv2.pointPolygonTest(c,c0,False)>=0
        s=a*(1 if ins else .3)
        if s>bs:bs=s;best=c
    if best is None: return None
    hull=cv2.convexHull(best);peri=cv2.arcLength(hull,True);quad=None;eps=.01
    while eps<=.12:
        ap=cv2.approxPolyDP(hull,eps*peri,True)
        if len(ap)==4:quad=ap.reshape(4,2).astype(np.float32);break
        if len(ap)<4:break
        eps+=.005
    if quad is None:
        quad=cv2.boxPoints(cv2.minAreaRect(hull))
    return order_points(quad/scale)

def old_gutter(spread):
    w=spread.shape[1]
    g=cv2.cvtColor(spread,cv2.COLOR_BGR2GRAY).astype(np.float32)
    col=g.mean(0)
    k=w//40; k=k+1 if k%2==0 else k
    col=cv2.GaussianBlur(col.reshape(1,-1),(max(k,5),1),0).ravel()
    lo,hi=int(w*.42),int(w*.58)
    bi=lo+int(col[lo:hi+1].argmin());bv=col[bi]
    med=np.sort(col)[w//2]
    return w//2 if bv>0.97*med else bi
