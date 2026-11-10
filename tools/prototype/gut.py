import numpy as np, cv2, random, itertools
import synth, edge
from synth import WORDS

def flat(seed,inner,blur_s,depth,off,FW=1400,FH=1000,sh=0):
    rng=random.Random(seed)
    img=np.full((FH,FW,3),(228,232,236),np.uint8); tm=np.zeros((FH,FW),np.uint8)
    gx=FW//2+off
    for (xa,xb) in [(95,gx-inner),(gx+inner,FW-95)]:
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
    X=np.arange(FW)[None,:,None].astype(np.float32)
    sx=gx+sh
    shadow=1-depth*np.exp(-((X-sx)/55.0)**2)-0.10*np.exp(-((X-sx)/200.0)**2)
    blur=cv2.GaussianBlur(img,(0,0),blur_s)
    wmix=np.exp(-((X-sx)/(130.0))**2)
    img=(img*(1-wmix)+blur*wmix)*shadow
    tL=tm.copy();tL[:,gx:]=0;tR=tm.copy();tR[:,:gx]=0
    return np.clip(img,0,255).astype(np.uint8),tL,tR,gx

print("inner blur depth shadowOffset | old: leak% cutPx | new: leak% cutPx risky")
agg={'old':[0,0,0],'new':[0,0,0]}
n=0
for inner,bs,dp,sh in itertools.product([85,50],[3.5,7],[0.35,0.7],[0,45,-45,70,-70]):
    img,tL,tR,gx=flat(n+5,inner,bs,dp,0,sh=sh); n+=1
    img=cv2.GaussianBlur(img,(0,0),0.8)
    line=f"{inner:3d} {bs:3.1f} {dp:.2f} {sh:4d} |"
    for m in ['old','new']:
        if m=='old': xc,risky=synth.old_gutter(img),False
        else: xc,risky=edge.find_cut(img)
        leak=((tR[:,:xc]>0).sum()+(tL[:,xc:]>0).sum())
        cut=((tL|tR)[:,max(0,xc-1):xc+2]>0).sum()
        tot=((tL|tR)>0).sum()
        agg[m][0]+=leak/tot*100;agg[m][1]+=cut; agg[m][2]+= (leak>0 or cut>0)
        line+=f" {m}: {leak/tot*100:5.2f} {cut:4d}{' R' if risky else '  '} |"
    print(line)
for m in agg: print(m,'mean leak%.3f'%(agg[m][0]/n),'mean cutTextPx %.1f'%(agg[m][1]/n),'cases with any text hit',agg[m][2],'/',n)
