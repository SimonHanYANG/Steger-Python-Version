"""Independent pixel-area integration of known continuous test objects.

This module does not use the detector's derivative kernels or bias inversion.
Pixel centers are integers. No thresholded drawing is used as ground truth.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def _profile(d,w,a,model):
    if model=="bar-shaped":return np.where(d<-w,0.0,np.where(d<=w,1.0,a))
    if model=="gaussian":shape=np.exp(-0.5*(d/w)**2)
    elif model=="parabolic":shape=np.maximum(0,1-(d/w)**2)
    else:raise ValueError(model)
    return np.where(d<=0,shape,a+(1-a)*shape)


def _pixel_cdf(t,n):
    a,b=sorted(np.abs(n),reverse=True)
    if b<1e-12:return np.clip(t/a+0.5,0,1)
    def sq(x):return np.maximum(x,0)**2
    s=(a+b)/2;z=(a-b)/2
    out=(sq(t+s)-sq(t+z)-sq(t-z)+sq(t-s))/(2*a*b)
    return np.where(t<=-s,0,np.where(t>=s,1,out))


def straight_line(shape=(96,96), *, angle_deg=0.0, offset=0.0, half_width=1.5,
                  asymmetry=0.0, model="bar-shaped", contrast=150.0,
                  background=30.0, polarity="light", noise=0.0, seed=0, quadrature=24,
                  quantize=False):
    """Infinite line; angle is the normal's angle from +row toward +column."""
    rr,cc=np.indices(shape,dtype=float)
    angle=np.deg2rad(angle_deg)
    n=np.array([np.cos(angle),np.sin(angle)])
    center=(np.asarray(shape)-1)/2+offset*n
    d=(rr-center[0])*n[0]+(cc-center[1])*n[1]
    if model=="bar-shaped":
        image=_pixel_cdf(d+half_width,n)-(1-asymmetry)*_pixel_cdf(d-half_width,n)
    else:
        nodes,weights=leggauss(quadrature);image=np.zeros(shape)
        for u,wu in zip(nodes/2,weights/2):
            for v,wv in zip(nodes/2,weights/2):
                image+=wu*wv*_profile(d+u*n[0]+v*n[1],half_width,asymmetry,model)
    image=background+contrast*image if polarity=="light" else background+contrast*(1-image)
    if noise:image+=np.random.default_rng(seed).normal(0,noise,shape)
    if quantize:image=np.clip(np.rint(image),0,255).astype(np.uint8)
    return image,{"normal":n,"center":center,"half_width":half_width,"angle_deg":angle_deg}


def circular_line(shape=(128,128), *, radius=35.0, half_width=1.5,
                  model="gaussian", asymmetry=0.0, quadrature=16, noise=0.0, seed=0):
    center=(np.asarray(shape)-1)/2
    rr,cc=np.indices(shape,dtype=float)
    nodes,weights=leggauss(quadrature);image=np.zeros(shape)
    for u,wu in zip(nodes/2,weights/2):
        for v,wv in zip(nodes/2,weights/2):
            d=np.hypot(rr+u-center[0],cc+v-center[1])-radius
            image+=wu*wv*_profile(d,half_width,asymmetry,model)
    image=30+150*image
    if noise:image+=np.random.default_rng(seed).normal(0,noise,shape)
    return image,{"center":center,"radius":radius,"half_width":half_width}

