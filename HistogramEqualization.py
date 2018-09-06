import cv2
import numpy as np
import matplotlib.pyplot as plt

def HistEqualize(grayimg):
    L=256
    unique,count=np.unique(grayimg,return_counts=True)
    cdf=np.cumsum(count)
    total_pixels=cdf[-1]
    equalized=np.uint8((L-1)*cdf/total_pixels)
    img_array=grayimg.ravel()
    for prev,new in zip(unique,equalized):
        img_array[img_array==prev]=new
        
    return img_array.reshape(grayimg.shape)
    
def ESIHE(grayimg):
    L=256                                               #No of gray levels in the output
    unique,count=np.unique(grayimg,return_counts=True)  #Forming the histogram.
                                                        #unique is unique gray levels and count is no of pixels in each  bin
    exposure=np.dot(unique,count)/(L*np.sum(count))     
    Xa=round((1-exposure)*L)                            #Gray-level threshold
    Tc=round(np.sum(count)/L)                           #Threshold for reducing over equalization
    count[count>Tc]=Tc                                  #Thresholding    
    Il=count[unique<=Xa].copy()                         #Under exposed image's histogram
    Iu=count[unique>Xa].copy()                          #Under exposed image's histogram
    
    #Calculating CDFs for under and over exposed image
    Cl=np.cumsum(Il/np.sum(Il))
    Cu=np.cumsum(Iu/np.sum(Iu))
    
    #Calculating new gray levels
    Fl=np.round((Xa+1)*Cl)
    Fu=np.round(Xa+(L-Xa-1)*Cu)
    count[unique<=Xa]=Fl
    count[unique>Xa]=Fu
    
    image_array=grayimg.ravel()
    for prev,new in zip(unique,np.uint8(count)):
        image_array[image_array==prev]=new
        
    return image_array.reshape(grayimg.shape)
  
def BBHE(grayimg):
    unique,count=np.unique(grayimg,return_counts=True)  #Forming the histogram.
                                                        #unique is unique gray levels and count is no of pixels in each  bin
         
    Xa=round(np.dot(unique,count)/np.sum(count))                                #Gray-level threshold    
    print("Xa:",Xa)
    Il=count[unique<=Xa].copy()                         #Under exposed image's histogram
    Iu=count[unique>Xa].copy()                          #Under exposed image's histogram
    
    #Calculating CDFs for under and over exposed image
    Cl=np.cumsum(Il/np.sum(Il))
    Cu=np.cumsum(Iu/np.sum(Iu))
    
    #Calculating new gray levels
    X0=unique[0]
    Fl=np.round(X0+(Xa-X0)*Cl)
    Fu=np.round(Xa+1+(unique[-1]-Xa-1)*Cu)
    count[unique<=Xa]=Fl
    count[unique>Xa]=Fu
    
    image_array=grayimg.ravel()
    for prev,new in zip(unique,np.uint8(count)):
        image_array[image_array==prev]=new
        
    return image_array.reshape(grayimg.shape)    

####### Demo #######    
    
imagename="View.jpg"
image=cv2.imread(imagename,0)
cv2.imshow("Original",image)
plt.hist(image.ravel(),256,[0,256]); plt.show()

equal=BBHE(image.copy())
plt.hist(equal.ravel(),256,[0,256]); plt.show()

cv2.imshow("Equalized",equal)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()

