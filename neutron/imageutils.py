# Author: Andreas Kaestner, Paul Scherrer Institute



import numpy as np
import scipy.ndimage.filters as flt2
from scipy import ndimage
import matplotlib.pyplot as plt
from tqdm import tqdm
import skimage.filters as flt
import skimage.morphology as morph
from timeit import default_timer as timer
#import skimage.morphology.greyreconstruct as rec

def _spotclean(img,size=5,threshold=0.95):
    fimg=flt2.median_filter(img,size)
    dimg=np.abs(img-fimg)
    
    nbins=256
    h,dx=np.histogram(dimg, bins=nbins, density=True)
    ch=np.cumsum(h)
    ch=ch/np.max(ch)
    th=np.min(dx[np.where(ch>threshold)])
    mask=dimg<th
    
    cimg=mask*img+(1-mask)*fimg
    
    return cimg

def spotclean(img,size=5,threshold=0.95) :
    fimg = []
    if len(img.shape)==2 :
        fimg =  _spotclean(img,size,threshold)
    else :
        fimg = np.zeros(img.shape)
        
        for idx in range(img.shape[0]) :
            fimg[idx]=_spotclean(img[idx],size,threshold)
            
    return fimg


def fill_spots(img,size=5) :
    # To be applied to each projection. 
    med = flt.median(img,footprint=np.ones([size,1])) # Moving Median filter in x direction
    med = flt.median(med,footprint=np.ones([1,size])) # Moving median filter in y-dimension
    fm = img.copy()
    fm[1:-2,1:-2] = med[1:-2,1:-2]
    fm = np.maximum(fm,img) # Take the pointwise maximum of th filtered and original image.
    res=morph.reconstruction(fm,img,method='erosion') # See: Robinson, “Efficient morphological reconstruction: a downhill filter”, Pattern Recognition Letters 25 (2004) 1759-1767. or
    # https://scikit-image.org/docs/stable/api/skimage.morphology.html#r4e1a5d6f491d-1
    return res


def fill_spots2(img,size=5) :
    med = morph.dilation(img,footprint=np.ones([size,size]))
    fm = img.copy()
    fm[1:-2,1:-2] = med[1:-2,1:-2]
    
    res=morph.reconstruction(fm,img,method='erosion')
    
    return res

def _morph_spot_clean(img,th_peaks=0.95,th_holes=0.95,method=0,size=5) :
    # Processes each projection.
    if method==0 : # Apply median filter to each dimension seperately
        fp=-fill_spots(-img,size=size)
        fh=fill_spots(img,size=size)
    else: # Apply simultanious median filter
        fp=-fill_spots2(-img)
        fh=fill_spots2(img)
    
    dh=np.abs(img-fh)
    dp=np.abs(img-fp)
    
    hh,ah=np.histogram(dh.ravel(),bins=1024);
    hp,ap=np.histogram(dp.ravel(),bins=1024);
    chh=np.cumsum(hh)
    chp=np.cumsum(hp)
    res=img.copy()
    if (th_holes < 1) :
        thh=ah[np.argmax(th_holes<chh/chh[-1])]
        res[thh<dh]=fh[thh<dh]
    if (th_peaks < 1) :
        thp=ap[np.argmax(th_peaks<chp/chp[-1])]
        res[thp<dp]=fp[thp<dp]
    
    return res

def morph_spot_clean(img,th_peaks=0.95,th_holes=0.95,method=0,size=5) :
    # Loops over each projection
    if (len(img.shape) == 2 ) :
        res = _morph_spot_clean(img,th_peaks,th_holes,method,size)
    else :
        res = np.zeros_like(img)
        for idx in range(img.shape[0]) :
            res[idx] = _morph_spot_clean(img[idx],th_peaks,th_holes,method,size)
    
    return res




def linepattern(segmentlength,f):
    """ Generates a 1D bilevel test pattern with increasing frequency
    
    Arguments:
    segmentlength -- number of pixels for each frequency
    f -- list of periods
    
    """
    
    x=np.arange(0.0,segmentlength)
    xx=np.ones(int(np.floor(segmentlength/3)))
    for ff in f :
        xx=np.append(xx,np.mod(np.floor(x/ff),2))
    
    return xx

def linepattern2d(segmentwidth,segmentheight,f,margin=False) :
    """ Generates a 2D bilevel line pattern with increasing frequency
    
    Arguments:
    segmentwidth -- number of pixels for each frequency
    segmentheight -- number of pixels for each line 
    f -- list of periods
    
    """
    
    
    x=linepattern(segmentwidth,f)
    
    if (margin==True) :
        y=np.stack([np.ones(len(x)),x,np.ones(len(x))])
    else :
        y=[x,x]
    
    return np.repeat(y,segmentheight,axis=0)

def contraststeps(neutrons=100,scalex=20,scaley=50) :
    """ Generates a constrst step wedge
      
    """
    img=np.array([[  1,   1,   1,   1,   1,   1,   1,   1,   1,  1,    1,   1, 1.0],
         [1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 1.0, 1.0],
         [  1,   1,   1,   1,   1,   1,   1,   1,   1,  1,    1,   1, 1.0]])
    
    img=np.repeat(img,scalex,axis=1)
    img=np.repeat(img,scaley,axis=0)
    
    return img

def neutronimage(img,sigma,photons,photonstrength):
    nimg=np.random.poisson(img)
    nimg=nimg-img
    
    nimg=img+ndimage.gaussian_filter(nimg,sigma)
    if (0.0<photons) :
        nimg=np.random.poisson(nimg*photons)/photons*photonstrength
    
    return nimg

def generatespots(size, fraction, width, amplitude,bias) :
    if (len(size)==1) :
        img=(np.random.uniform(0.0,1.0,(size,size))<fraction)*1.0
    else :
        img=(np.random.uniform(0.0,1.0,size)<fraction)*1.0
    img=ndimage.filters.gaussian_filter(img,sigma=width)
    img=(img/img.max())*amplitude+bias
    
    return img

def buildimagestack(size,nimg,ncount) :
    imgs=np.ones(shape=[size,size,nimg])*ncount
    for i in np.arange(nimg) :
        imgs[:,:,i] = neutronimage(imgs[:,:,i],1.0,2.0,1.0)
    
    return imgs

def singleimage(size,ncount) :
    img=np.ones(shape=[size,size])*ncount
    img = neutronimage(img,1.0,2.0,1.0)
    
    return img

def averageimage(imgs, axis=0) :
    img=imgs.mean(axis=axis)
    
    return img

def medianimage(imgs,axis=0) :
    img=np.median(imgs,axis=axis)
    
    return img

def weightedaverageimage(imgs,size) :
    dims=imgs.shape
    w=np.zeros(imgs.shape)
    M=size**2
    print('M=',M)
    fig,ax = plt.subplots(dims[0],4,figsize=(12,30))
    ax = ax.ravel()
    for i in np.arange(dims[0]) :
        print('i=',i)
        f=ndimage.filters.uniform_filter(imgs[i], size=size)*M
        f2=ndimage.filters.uniform_filter(imgs[i]**2, size=size)*M
        
        sigma=(1/(M-1)*(f2-(f**2)/M))**2
        w[i]=1.0/sigma
        ax[i*4].imshow(f)
        ax[i*4+1].imshow(f2)
        ax[i*4+2].imshow(sigma)
        ax[i*4+3].imshow(w[i])

    wsum=w.sum(axis=0)
    for i in np.arange(dims[0]) :
        w[i]=w[i]/wsum

    imgs=w*imgs
    img=imgs.sum(axis=0)
    
    return img



def periodicSpotPattern(dims, width, distance, amplitude) :
    x,y = np.meshgrid(range(0,dims[1]),range(0,dims[0]))
    dots=amplitude*(np.mod(x,distance)==0)*(np.mod(y,distance)==0)
    fdots=ndimage.filters.gaussian_filter(dots,width)
    m=np.max(fdots)
    fdots=amplitude/m*fdots
    
    return fdots    

def SNR(data,reference) :
    MSE = np.mean((data-reference)**2)
    s=np.sqrt(MSE)
    m=np.mean(reference)
    
    return s