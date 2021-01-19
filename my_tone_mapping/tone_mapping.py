import os
import sys
import cv2
import numpy as np 

def log_(img):
    log_img = np.log(np.maximum(img, np.ones(img.shape) * (1e-4)))
    return log_img

def mapLuminace(img, gray_img, new_gray_img, saturation):
    # ret = np.zeros(img.shape)
    # for i in range(3):
    #     ret[:,:,i] = (img[:,:,i] / gray_img) * new_gray_img
    # return ret
    channels = img / np.expand_dims(gray_img, axis=2)
    channels = np.power(channels, saturation)
    channels = channels *np.expand_dims(new_gray_img, axis=2)

    return channels

def toneMappingDurand(img, gamma = 1.0, contrast = 4.0, saturation = 1.0, sigma_space = 2.0, sigma_color = 2.0):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    log_img = log_(gray_img)
    log_img = log_img.astype(np.float32)
    print(log_img.dtype)
    map_img = cv2.bilateralFilter(log_img, -1, sigma_color, sigma_space)
    print(map_img.dtype)
    scale = contrast / (np.max(map_img) - np.min(map_img))
    print(scale)
    new_gray_img = np.exp(map_img * (scale - 1.0) + log_img)
    ret = mapLuminace(img, gray_img, new_gray_img, saturation)
    ret = np.power(ret, 1.0/gamma)
    return ret
# 
# hdr_img_path = "vinesunset.hdr"
hdr_img_path = "GoldenGate_2k.hdr"
# hdr_img_path = "res.hdr"

hdr_img = cv2.imread(hdr_img_path, flags=cv2.IMREAD_ANYDEPTH)[5:-5,5:-5,:]

# W, H, C = hdr_img.shape
# sigma_space = 0.02 * max(W, H)
# sigma_color = 0.04

# ldr_img = toneMapping(hdr_img, 1, np.log2(5) , sigma_space, sigma_color)
ldr_img = toneMappingDurand(hdr_img, 2.2)
ldr_img = np.clip(ldr_img*255, 0, 255).astype('uint8')
cv2.imwrite("ldr_img.jpg", ldr_img)

# opencv implementation, needs opencv version==3.4.2
# tonemapDurand = cv2.createTonemapDurand(2.2)
# ldrDurand = tonemapDurand.process(hdr_img)
# ldrDurand = np.clip(ldrDurand*255, 0, 255).astype('uint8')
# # ldrDurand = 3 * ldrDurand
# cv2.imwrite("opencv_ldr.jpg", ldrDurand )





