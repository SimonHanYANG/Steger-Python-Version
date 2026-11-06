# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Experimental compact mask-only filtering and normalized cubic sampling.

Only the M supplied gray values are image samples. The integer index map is
mask metadata, with -1 for absent pixels; outside image values are never read.
Cubic basis formulas follow _spline_native.pyx; see THIRD_PARTY_NOTICES.md.
"""
import numpy as np
from libc.math cimport floor, fabs, isfinite


def gaussian_fields(const double[::1] gray, const Py_ssize_t[:, ::1] pixels,
                    const double[::1] g0, const double[::1] g1, const double[::1] g2):
    cdef Py_ssize_t n=gray.shape[0], radius=g0.shape[0]//2
    if pixels.shape[0]!=n or pixels.shape[1]!=2 or g1.shape[0]!=g0.shape[0] or g2.shape[0]!=g0.shape[0]:
        raise ValueError('Invalid compact Gaussian inputs')
    result=np.empty((5,n),dtype=np.float64)
    cdef double[:,::1] out=result
    cdef Py_ssize_t i,j,dr,dc,k
    cdef double a[6]
    cdef double b[6]
    cdef double weights[6]
    cdef double value,s,sr,sc
    with nogil:
        for i in range(n):
            for k in range(6):
                a[k]=0.;b[k]=0.
            for j in range(n):
                dr=pixels[i,0]-pixels[j,0]
                dc=pixels[i,1]-pixels[j,1]
                if dr < -radius or dr > radius or dc < -radius or dc > radius:
                    continue
                dr+=radius;dc+=radius
                weights[0]=g0[dr]*g0[dc]
                weights[1]=g1[dr]*g0[dc]
                weights[2]=g0[dr]*g1[dc]
                weights[3]=g2[dr]*g0[dc]
                weights[4]=g1[dr]*g1[dc]
                weights[5]=g0[dr]*g2[dc]
                value=gray[j]-gray[i]
                for k in range(6):
                    a[k]+=value*weights[k]
                    b[k]+=weights[k]
            s=a[0]/b[0]
            sr=(a[1]-s*b[1])/b[0]
            sc=(a[2]-s*b[2])/b[0]
            out[0,i]=sr;out[1,i]=sc
            out[2,i]=(a[3]-2.*sr*b[1]-s*b[3])/b[0]
            out[3,i]=(a[4]-sr*b[2]-sc*b[1]-s*b[4])/b[0]
            out[4,i]=(a[5]-2.*sc*b[2]-s*b[5])/b[0]
    return result


cdef inline void basis(double x, double* w) noexcept nogil:
    cdef double y=x-floor(x), z=1.-y
    w[1]=(y*y*(y-2.)*3.+4.)/6.
    w[2]=(z*z*(z-2.)*3.+4.)/6.
    w[0]=z*z*z/6.
    w[3]=1.
    w[3]-=w[0];w[3]-=w[1];w[3]-=w[2]


cdef inline bint inside(const Py_ssize_t[:,::1] index, double r, double c) noexcept nogil:
    cdef Py_ssize_t rr=<Py_ssize_t>floor(r+.5), cc=<Py_ssize_t>floor(c+.5)
    return 0<=rr<index.shape[0] and 0<=cc<index.shape[1] and index[rr,cc]>=0


cdef inline void sample(const double[:,::1] co, const Py_ssize_t[:,::1] index,
                        double r, double c, double* out) noexcept nogil:
    cdef double rw[4]
    cdef double cw[4]
    cdef double weight,denom=0.
    cdef Py_ssize_t sr=<Py_ssize_t>floor(r)-1, sc=<Py_ssize_t>floor(c)-1
    cdef Py_ssize_t a,b,k,j,rr,cc
    basis(r,rw);basis(c,cw)
    for k in range(5):out[k]=0.
    for a in range(4):
        rr=sr+a
        if rr<0 or rr>=index.shape[0]:continue
        for b in range(4):
            cc=sc+b
            if cc<0 or cc>=index.shape[1]:continue
            j=index[rr,cc]
            if j<0:continue
            weight=rw[a]*cw[b]
            denom+=weight
            for k in range(5):out[k]+=weight*co[j,k]
    for k in range(5):out[k]/=denom


def sample_fields(const double[:,::1] co, const Py_ssize_t[:,::1] index,
                  const double[:,::1] points):
    if co.shape[1]!=5 or points.shape[1]!=2:raise ValueError('Invalid compact spline inputs')
    cdef Py_ssize_t i,k,n=points.shape[0]
    result=np.empty((5,n),dtype=np.float64)
    cdef double[:,::1] out=result
    cdef double values[5]
    for i in range(n):
        if not isfinite(points[i,0]) or not isfinite(points[i,1]) or not inside(index,points[i,0],points[i,1]):
            raise ValueError('Query must remain inside the mask')
    with nogil:
        for i in range(n):
            sample(co,index,points[i,0],points[i,1],values)
            for k in range(5):out[k,i]=values[k]
    return result


def refine_points(const double[:,::1] co, const Py_ssize_t[:,::1] index,
                  double[:,::1] points, const double[:,::1] normals,
                  const Py_ssize_t[:,::1] origins):
    if co.shape[1]!=5 or points.shape[1]!=2 or normals.shape[0]!=points.shape[0] or origins.shape[0]!=points.shape[0]:
        raise ValueError('Invalid compact refinement inputs')
    cdef Py_ssize_t i,iteration,n=points.shape[0]
    cdef double d[5]
    cdef double nr,nc,denom,step,r,c,max_step
    for i in range(n):
        if not isfinite(points[i,0]) or not isfinite(points[i,1]) or not inside(index,points[i,0],points[i,1]):
            raise ValueError('Initial point must remain inside the mask')
    with nogil:
        for iteration in range(6):
            max_step=0.
            for i in range(n):
                sample(co,index,points[i,0],points[i,1],d)
                nr=normals[i,0];nc=normals[i,1]
                denom=d[2]*nr*nr+2.*d[3]*nr*nc+d[4]*nc*nc
                step=-(d[0]*nr+d[1]*nc)/denom if fabs(denom)>1e-14 else 0.
                if step<-.25:step=-.25
                elif step>.25:step=.25
                r=points[i,0]+step*nr;c=points[i,1]+step*nc
                if fabs(r-origins[i,0])<=1.5 and fabs(c-origins[i,1])<=1.5 and inside(index,r,c):
                    points[i,0]=r;points[i,1]=c
                if fabs(step)>max_step:max_step=fabs(step)
            if max_step<1e-8:break
    return np.asarray(points)
