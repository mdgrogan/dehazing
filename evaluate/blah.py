import cv2
import numpy as np
from skimage import data, img_as_float
from skimage.measure import compare_mse as mse 
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim 


def CLAHE(bgr, lim, gs):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=lim, tileGridSize=(gs, gs))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

label = cv2.imread("img/1-clear.png")
I = cv2.imread("img/1_AOD-Net.png")
J = CLAHE(I, 0.8, 18)
flabel = img_as_float(label)
fI = img_as_float(I)
fJ = img_as_float(J)

mseI = mse(fI, flabel)
psnrI = psnr(fI, flabel)
ssimI = ssim(fI, flabel, multichannel=True)

mseJ = mse(fJ, flabel)
psnrJ = psnr(fJ, flabel)
ssimJ = ssim(fJ, flabel, multichannel=True)

print("{0:f}, {1:f}, {2:f}".format(mseI, psnrI, ssimI))
print("{0:f}, {1:f}, {2:f}".format(mseJ, psnrJ, ssimJ))




#lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
#lab_planes = cv2.split(lab)
#range_ = np.max(lab_planes[0]) - np.min(lab_planes[0])
#mean_ = np.mean(lab_planes[0])
#var_ = np.var(lab_planes[0])
#print(range_)
#print(mean_)
#print(var_)

