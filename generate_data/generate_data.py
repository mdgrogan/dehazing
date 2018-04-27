import os
import re
import numpy as np
import math
import caffe
import cv2
from skimage import data, img_as_float
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim 


def EditFcnProto(templateFile, height, width):
    with open(templateFile, 'r') as ft:
        template = ft.read()
    #print(templateFile)
    outFile = 'DeployT.prototxt'
    with open(outFile, 'w') as fd:
        fd.write(template.format(height=height,width=width))

def runAOD(img):
    model='AOD_Net.caffemodel'

    net = caffe.Net('DeployT.prototxt', model, caffe.TEST)
    batchdata = []
    data = img/255.0
    data = data.transpose((2, 0, 1))
    batchdata.append(data)
    net.blobs['data'].data[...] = batchdata

    net.forward();

    data = net.blobs['sum'].data[0]
    data = data.transpose((1, 2, 0))
    data = data*255.0
    return data.astype(np.uint8)


def CLAHE(bgr, clip, tile):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return bgr

def meanBrightness(im):
    #im = cv2.imwrite(im)
    im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_mean = cv2.mean(im_g)
    return im_mean

def globalVar(im):
    im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_std_dev = cv2.meanStdDev(im_g)
    im_var = math.sqrt(im_std_dev[1])
    return im_var

def histEntropy(im):
    im_g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist(im,[0],None,[256],[0,256])
    hist = hist_gray/im_g.size
    hist_log = cv2.log(hist)
    
    n=0
    while n<256:
        if math.isinf(hist_log[n]):
            hist_log[n]=0
        n+=1
    
    entropy = -1*cv2.sumElems(np.multiply(hist,hist_log))[0]
    return entropy

def contrastRatio(im):
    im_ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
    im_height, im_width, im_ch = im.shape
    
    histY = cv2.calcHist(im_ycc,[0],None,[256],[0,256])
    histCb = cv2.calcHist(im_ycc,[1],None,[256],[0,256])
    histCr = cv2.calcHist(im_ycc,[2],None,[256],[0,256])
    
    n=0
    f0=0; f1=0; f2=0;
    while n<256:
        f0 += histY[n] * n
        f1 += histCb[n] * n
        f2 += histCr[n] * n
        n+=1
    
    pixelCount = im_height * im_width
    average_Y = f0/pixelCount
    average_Cb = f1/pixelCount
    average_Cr = f2/pixelCount
    
    n=0; distance_Y=0; distance_Cr=0; distance_Cb=0
    
    while n<256:
        distance_Y += abs(average_Y - histY[n])
        distance_Cr += abs(average_Cb - histCb[n])
        distance_Cb += abs(average_Cr - histCr[n])
        n+=1
        
    contrast = distance_Y + distance_Cr + distance_Cb
    return contrast

def imageMean(im):
    im_ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)
    im_Y, im_Cr, im_Cb = cv2.split(im_ycc)
    im_Ytuple = cv2.mean(im_Y)
    im_Ymean = im_Ytuple[0]
    return im_Ymean

def generate_OTS_data(limit):
    OTS_train_dir = '/home/grogan/Storage/CSCE633_Project_data/train'
    OTS_clear_dir = '/home/grogan/Storage/CSCE633_Project_data/clear_images'
    caffe.set_mode_cpu()
    imagesnum = 0
    for imagename in os.listdir(OTS_train_dir):
        clear_imagename = (imagename.split('_'))[0] + '.jpg'
        if (os.path.isfile(r'{}/{}'.format(OTS_train_dir, imagename) == False)):
            continue
        if (os.path.isfile(r'{}/{}'.format(OTS_clear_dir, clear_imagename) == False)):
            continue

        imagesnum += 1
        #print('{}/{}'.format(OTS_train_dir, imagename))
        haze = cv2.imread('{}/{}'.format(OTS_train_dir, imagename))
        clear = cv2.imread('{}/{}'.format(OTS_clear_dir, clear_imagename))
        height = haze.shape[0]
        width = haze.shape[1]

        templateFile = 'test_template.prototxt'
        EditFcnProto(templateFile, height, width)

        data = runAOD(haze)

        m_b = meanBrightness(data)
        g_v = globalVar(data)
        h_e = histEntropy(data)
        c_r = contrastRatio(data)
        i_m = imageMean(data)

        f_data = img_as_float(data)
        f_clear = img_as_float(clear)
        psnr_data = psnr(f_data, f_clear)
        ssim_data = ssim(f_data, f_clear, multichannel=True)
        
        # clip tile psnr ssim
        bestPSNR = [0,0,0,0]
        bestSSIM = [0,0,0,0]

        clip = 0.1
        for ii in range(20):
            tile = 1
            for jj in range(25):
                output = CLAHE(data, clip, tile)
                f_output = img_as_float(output)
                
                psnr_output = psnr(f_output, f_clear)
                ssim_output = ssim(f_output, f_clear, multichannel=True)
                print("img:{0}, clip:{1:.1f}, grid:{2}".format(imagesnum, clip, tile))
                print("psnr:{0:4f}, ssim:{1:4f}".format(psnr_output, ssim_output))
                if (psnr_output > bestPSNR[2]):
                    bestPSNR = [clip, tile, psnr_output, ssim_output]
                    print("new best psnr:{0:4f}, ssim:{1:4f}".format(psnr_output, ssim_output))
                if (ssim_output > bestSSIM[3]):
                    bestSSIM = [clip, tile, psnr_output, ssim_output]
                    print("new best ssim:{0:4f}, ssim:{1:4f}".format(psnr_output, ssim_output))

                tile += 1
            clip += 0.1

        f = open('bestPSNR.txt', 'a')
        f.write("{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} "
                "{5:.1f} {6:d} {7:.4f} {8:.4f} {9:.4f} {10:.4f}\n"
                .format(m_b[0], g_v, h_e, c_r[0], i_m,
                        bestPSNR[0], bestPSNR[1], bestPSNR[2], bestPSNR[3],
                        psnr_data, ssim_data))
        f.close()
        f = open('bestSSIM.txt', 'a')
        f.write("{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} "
                "{5:.1f} {6:d} {7:.4f} {8:.4f} {9:.4f} {10:.4f}\n"
                .format(m_b[0], g_v, h_e, c_r[0], i_m,
                        bestSSIM[0], bestSSIM[1], bestSSIM[2], bestSSIM[3],
                        psnr_data, ssim_data))
        f.close()

        # some condition goes here if you don't want all 300,000 images
        if (imagesnum == limit):
            break
    
def generate_ITS_data(limit):
    ITS_train_dir = '/home/grogan/Storage/CSCE633_Project_data/its/hazy'
    ITS_clear_dir = '/home/grogan/Storage/CSCE633_Project_data/its/clear'
    caffe.set_mode_cpu()
    imagesnum = 0
    for imagename in os.listdir(ITS_train_dir):
        clear_imagename = (imagename.split('_'))[0] + '.png'
        if (os.path.isfile(r'{}/{}'.format(ITS_train_dir, imagename) == False)):
            continue
        if (os.path.isfile(r'{}/{}'.format(ITS_clear_dir, clear_imagename) == False)):
            continue

        imagesnum += 1
        print('{}/{}'.format(ITS_train_dir, imagename))
        print('{}/{}'.format(ITS_train_dir, clear_imagename))
        haze = cv2.imread('{}/{}'.format(ITS_train_dir, imagename))
        clear = cv2.imread('{}/{}'.format(ITS_clear_dir, clear_imagename))
        height = haze.shape[0]
        width = haze.shape[1]

        templateFile = 'test_template.prototxt'
        EditFcnProto(templateFile, height, width)

        data = runAOD(haze)

        m_b = meanBrightness(data)
        g_v = globalVar(data)
        h_e = histEntropy(data)
        c_r = contrastRatio(data)
        i_m = imageMean(data)

        f_data = img_as_float(data)
        f_clear = img_as_float(clear)
        psnr_data = psnr(f_data, f_clear)
        ssim_data = ssim(f_data, f_clear, multichannel=True)
        
        #maxPSNR = 0.0
        # clip tile psnr ssim
        bestPSNR = [0,0,0,0]
        #maxSSIM = 0.0
        bestSSIM = [0,0,0,0]

        clip = 0.1
        for ii in range(20):
            tile = 1
            for jj in range(25):
                output = CLAHE(data, clip, tile)
                f_output = img_as_float(output)
                
                psnr_output = psnr(f_output, f_clear)
                ssim_output = ssim(f_output, f_clear, multichannel=True)
                print("img:{0}, clip:{1:.1f}, grid:{2}".format(imagesnum, clip, tile))
                print("psnr:{0:4f}, ssim:{1:4f}".format(psnr_output, ssim_output))
                if (psnr_output > bestPSNR[2]):
                    #maxPSNR = psnr_output
                    bestPSNR = [clip, tile, psnr_output, ssim_output]
                    print("new best psnr:{0:4f}, ssim:{1:4f}".format(psnr_output, ssim_output))
                if (ssim_output > bestSSIM[3]):
                    bestSSIM = [clip, tile, psnr_output, ssim_output]
                    print("new best ssim:{0:4f}, ssim:{1:4f}".format(psnr_output, ssim_output))

                tile += 1
            clip += 0.1

        f = open('bestPSNR.txt', 'a')
        f.write("{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} "
                "{5:.1f} {6:d} {7:.4f} {8:.4f} {9:.4f} {10:.4f}\n"
                .format(m_b[0], g_v, h_e, c_r[0], i_m,
                        bestPSNR[0], bestPSNR[1], bestPSNR[2], bestPSNR[3],
                        psnr_data, ssim_data))
        f.close()
        f = open('bestSSIM.txt', 'a')
        f.write("{0:.4f} {1:.4f} {2:.4f} {3:.4f} {4:.4f} "
                "{5:.1f} {6:d} {7:.4f} {8:.4f} {9:.4f} {10:.4f}\n"
                .format(m_b[0], g_v, h_e, c_r[0], i_m,
                        bestSSIM[0], bestSSIM[1], bestSSIM[2], bestSSIM[3],
                        psnr_data, ssim_data))
        f.close()

        # some condition goes here if you don't want all 300,000 images
        if (imagesnum == limit):
            break

def main():
    f = open('bestPSNR.txt', 'w+')
    f.write("mB gV hE cR iM clip tile outputPSNR outputSSIM aodPSNR aodSSIM\n")
    f.close()
    f = open('bestSSIM.txt', 'w+')
    f.write("mB gV hE cR iM clip tile outputPSNR outputSSIM aodPSNR aodSSIM\n")
    f.close()
    generate_OTS_data(1)
    generate_ITS_data(1)


if __name__ == '__main__':
    main();
