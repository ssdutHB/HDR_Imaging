import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import os
import sys 
import math

folder_path = sys.argv[1]

def notEmpty(s):
    return s and s.strip()

def readImageInfo(folder_path, meta_data):
    f = open(meta_data)
    line = f.readline()
    line = f.readline()
    image_info = []
    while line:
        line = line.strip().split(" ")
        # line = list(filter(notEmpty, line))
        # filter the elements that is empty
        line = list(filter(lambda s: len(s)>0, line))
        print(line)
        name = line[0]
        exposure = float(line[1])
        # the meta data give the 1/shutter_speed
        shutter_speed = 1/float(line[-1])
        image_info.append([os.path.join(folder_path, name), exposure, shutter_speed])
        line = f.readline()
    images = [cv2.imread(info[0]) for info in image_info]
    return image_info, images

# find the value and index of max, mid and min pixels in a image patch
# return the index in the origial image
def selectPixels(a, ori_r, ori_c):
    H, W = a.shape
    fa = a.flatten()
    sorted_index = np.argsort(fa)
    min_index= sorted_index[0]
    max_index= sorted_index[-1]
    mid_index= sorted_index[len(fa) // 2]
    min_r = min_index // W
    min_c = min_index % W
    max_r = max_index // W
    max_c = max_index % W
    mid_r = mid_index // W
    mid_c = mid_index % W
    return [[ori_r + min_r, ori_c + min_c],[ori_r + mid_r, ori_c + mid_c],[ori_r + max_r, ori_c + max_c]]

# Given an image and pixel num 
# return selected pixels
def samplePixels(image, pixel_num):
    H, W = image.shape
    patch_num = int(np.sqrt(pixel_num // 3))
    H_patch_size = H // patch_num
    W_patch_size = W // patch_num
    ret = []
    print(H_patch_size, W_patch_size)
    for i in range(patch_num):
        for j in range(patch_num):
            img_patch = image[i * H_patch_size:(i + 1) * H_patch_size, j * W_patch_size: (j + 1) * W_patch_size]
            # cv2.imwrite(str(i) + "_" + str(j) + ".png", img_patch)
            selected_pos = selectPixels(img_patch, i * H_patch_size, j * W_patch_size)
            ret.append(selected_pos)
    ret = np.array(ret)
    ret = ret.reshape((np.prod(ret.shape[:-1]),ret.shape[-1]))
    return ret

def samplePixels2(image, number_of_samples_per_dimension):
    width = image.shape[0]
    height = image.shape[1]
    width_iteration = width / number_of_samples_per_dimension
    height_iteration = height / number_of_samples_per_dimension
    ret = []

    h_iter = 0
    for i in range(number_of_samples_per_dimension):
        w_iter = 0
        for j in range(number_of_samples_per_dimension):
            if math.floor(w_iter) < width and math.floor(h_iter) < height:
                ret.append([math.floor(w_iter), math.floor(h_iter)])
            w_iter += width_iteration
        h_iter += height_iteration
    return ret

def samplePixels3(images, pixel_num_per_dim):
    H, W = images[0].shape
    # patch_num = int(np.sqrt(pixel_num // 3))
    H_patch_size = H // pixel_num_per_dim
    W_patch_size = W // pixel_num_per_dim

    ret = np.zeros((pixel_num_per_dim * pixel_num_per_dim * 3, len(images)))
    for i in range(pixel_num_per_dim):
        for j in range(pixel_num_per_dim):
            for k in range(len(images)):
                # print(i * pixel_num_per_dim + j)
                # print(k)
                # print(i * H_patch_size)
                # print( j * W_patch_size)
                img_patch = images[0][i * H_patch_size:(i + 1) * H_patch_size, j * W_patch_size: (j + 1) * W_patch_size]
                selected_pos = selectPixels(img_patch, i * H_patch_size, j * W_patch_size)
                ret[i * pixel_num_per_dim *3 + 3 * j, k] = images[k][selected_pos[0][0], selected_pos[0][1]]
                ret[i * pixel_num_per_dim *3 + 3 * j + 1, k] = images[k][selected_pos[1][0], selected_pos[1][1]]
                ret[i * pixel_num_per_dim *3 + 3 * j + 2, k] = images[k][selected_pos[2][0], selected_pos[2][1]]

                # ret[i * pixel_num_per_dim + j, k] = images[k][i * H_patch_size, j * W_patch_size]
    return ret


def samplePixels4(images, pixel_num_per_dim):
    H, W = images[0].shape
    # patch_num = int(np.sqrt(pixel_num // 3))
    H_patch_size = H // pixel_num_per_dim
    W_patch_size = W // pixel_num_per_dim

    ret = np.zeros((pixel_num_per_dim * pixel_num_per_dim, len(images)))
    for i in range(pixel_num_per_dim):
        for j in range(pixel_num_per_dim):
            for k in range(len(images)):

                ret[i * pixel_num_per_dim + j, k] = images[k][i * H_patch_size, j * W_patch_size]
    return ret



def gSolve(Z, B, smooth_lambda, weights):
    max_value = 255
    min_value = 0
    n = 256
    
    A = np.zeros((Z.shape[0] * Z.shape[1] + n + 1, n + Z.shape[0]))
    b = np.zeros((A.shape[0], 1))

    k = 0
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            wij = weights[Z[i,j]]
            A[k, Z[i, j]] = wij
            A[k, n+i] = -1*wij
            b[k,0] = wij * B[j]
            k = k + 1
    
    # fix the curve by setting its middle value t0 0
    A[k, 128] = 1
    k = k + 1

    for i in range(n-2):
        A[k, i] = smooth_lambda * weights[i+1]
        A[k, i+1] = -2 * smooth_lambda * weights[i+1]
        A[k, i+2] = smooth_lambda * weights[i+1]
        k = k + 1
    res = np.linalg.lstsq(A, b)[0]
    g = res[:n]
    lE = res[n:]

    return g, lE


# Implementation of paper's Equation(3) with weight
def response_curve_solver(Z, B, l, w):
    n = 256
    A = np.zeros(shape=(np.size(Z, 0)*np.size(Z, 1)+n+1, n+np.size(Z, 1)), dtype=np.float32)
    b = np.zeros(shape=(np.size(A, 0), 1), dtype=np.float32)

    # Include the data−fitting equations
    k = 0
    for i in range(np.size(Z, 1)):
        for j in range(np.size(Z, 0)):
            z = int(Z[j][i])
            wij = w[z]
            A[k][z] = wij
            A[k][n+i] = -wij
            b[k] = wij*B[j]
            k += 1
    
    # Fix the curve by setting its middle value to 0
    A[k][128] = 1
    k += 1

    # Include the smoothness equations
    for i in range(n-1):
        A[k][i]   =    l*w[i+1]
        A[k][i+1] = -2*l*w[i+1]
        A[k][i+2] =    l*w[i+1]
        k += 1

    # Solve the system using SVD
    x = np.linalg.lstsq(A, b)[0]
    g = x[:256]
    lE = x[256:]

    return g, lE

def plotSingleCurve(g):
    g = g[:,0]
    x = np.linspace(0,255,256)
    plt.plot(g, x, 'b-', lw=3)
    plt.show()

def GB(value):
    return g_b[value]


def GG(value):
    return g_g[value]


def GR(value):
    return g_r[value]

def W(value):
    return weights[value]

def rebuildRadianceMap(images, B, F):
    FGB = np.vectorize(F)
    FW = np.vectorize(W)
    g_z = np.array([FGB(image) for image in images])
    w_z = np.array([FW(image) for image in images])
    BB = B.reshape((B.shape[0],1,1))
    BB = np.ones(g_z.shape) * BB
    # B.shape = g_z.shape
    tmp0 =  (g_z - BB)
    tmp1=  w_z * tmp0
    top = np.sum(tmp1, axis = 0)
    bottom = np.sum(w_z, axis = 0)
    bottom[np.where(bottom==0)] = 1
    # print(tmp0[0][np.where(bottom==0)])
    top[np.where(bottom==0)] = np.min(tmp0)
    # print()
    ret = top / bottom
    print(ret.dtype)
    return ret
    

def save_hdr(hdr, filename):
    image = np.zeros((hdr.shape[0], hdr.shape[1], 3), 'float32')
    image[..., 0] = hdr[..., 2]
    image[..., 1] = hdr[..., 1]
    image[..., 2] = hdr[..., 0]

    f = open(filename, 'wb')
    f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
    header = '-Y {0} +X {1}\n'.format(image.shape[0], image.shape[1]) 
    f.write(bytes(header, encoding='utf-8'))

    brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
    mantissa = np.zeros_like(brightest)
    exponent = np.zeros_like(brightest)
    np.frexp(brightest, mantissa, exponent)
    scaled_mantissa = mantissa * 256.0 / brightest
    rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
    rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
    rgbe[...,3] = np.around(exponent + 128)

    rgbe.flatten().tofile(f)
    f.close()

def tonemapping(hdr_img):

    sigma_r = 0.4
    sigma_s = 0.02 * np.max(hdr_img.shape)
    f_kernel_size = 9
    # print(hdr_img.dtype)
    # print(hdr_img.shape)

    # intensity = np.average(hdr_img, axis=2)
    intensity = (hdr_img[:,:,0] + hdr_img[:,:,1] * 40 + hdr_img[:,:,2] * 20)/61.0

    # plt.subplot(241),plt.imshow(intensity),plt.title('intensity'),plt.xticks([]),plt.yticks([])
    intensity = np.expand_dims(intensity, axis=2)
    # print(intensity.shape)
    chrominace = hdr_img / intensity
    # print(chrominace.shape)

    log_intensity = np.log(intensity)
    B = cv2.bilateralFilter(log_intensity, f_kernel_size, sigma_r, sigma_s)
    B = np.expand_dims(B, axis=2)
    # plt.subplot(242),plt.imshow(B),plt.title('B'),plt.xticks([]),plt.yticks([])


    D = log_intensity - B
    s = np.log2(5) / (np.max(B) - np.min(B))

    B_ = B * s

    tmp = B_ + D

    O = np.array([[np.exp(tmp[i,j]) for j in range(tmp.shape[1]) ]for i in range(tmp.shape[0]) ])

    print(O.shape)
    print(chrominace.shape)

    ret = O * chrominace
    return ret



image_info, images = readImageInfo(folder_path, os.path.join(folder_path, "image_list.txt"))

print(images[0].shape)
images_b = [image[:,:,0] for image in images]
images_g = [image[:,:,1] for image in images]
images_r = [image[:,:,2] for image in images]
print(images_b[0].shape)
Z_b = samplePixels4(images_b, 20)
Z_g = samplePixels4(images_g, 20)
Z_r = samplePixels4(images_r, 20)
B_b = np.array([np.log(x[1]) for x in image_info])
weights = np.arange(256)
weights = [255 - x if x > 128 else x for x in weights]
smooth_lambda = 100
g_b, lE = gSolve(Z_b.astype(np.int64), B_b, smooth_lambda, weights)
g_g, lE = gSolve(Z_g.astype(np.int64), B_b, smooth_lambda, weights)
g_r, lE = gSolve(Z_r.astype(np.int64), B_b, smooth_lambda, weights)
# plt.figure(figsize=(10, 10))
# plt.plot(g_b, range(256), 'bx')
# plt.plot(g_g, range(256), 'gx')
# plt.plot(g_r, range(256), 'rx')
# plt.ylabel('pixel value Z')
# plt.xlabel('log exposure X')
# plt.savefig('response-curve.png')

radiance_b = rebuildRadianceMap(images_b, B_b, GB)
radiance_g = rebuildRadianceMap(images_g, B_b, GG)
radiance_r = rebuildRadianceMap(images_r, B_b, GR)


hdr = np.zeros((images[0].shape[0], images[0].shape[1], 3), 'float32')
hdr[:,:,0] = radiance_b
hdr[:,:,1] = radiance_g
hdr[:,:,2] = radiance_r
# vfunc = np.vectorize(lambda x:math.exp(x))

plt.subplot(121),plt.imshow(hdr),plt.title('hdr1'),plt.xticks([]),plt.yticks([])

hdr = np.exp(hdr)

res = tonemapping(hdr)

plt.subplot(122),plt.imshow(res),plt.title('res'),plt.xticks([]),plt.yticks([])
plt.show()

plt.figure(figsize=(12,8))
plt.imshow(np.log2(cv2.cvtColor(hdr, cv2.COLOR_BGR2GRAY)), cmap='jet')
plt.colorbar()
plt.savefig('radiance-map.png')




tonemapDrago = cv2.createTonemapDrago(1.0, 0.7)
print(np.max(hdr), np.min(hdr))
ldrDrago = tonemapDrago.process(hdr)
ldrDrago = 3 * ldrDrago
cv2.imwrite("ldr-Drago.jpg", ldrDrago * 255)


# save_hdr(hdr, 'res.hdr')
print(np.max(hdr), np.min(hdr))
cv2.imwrite('res.hdr', hdr)
# import nibabel as nib
# hdr = nib.Nifti1Image(hdr, affine=np.eye(4))
# nib.nifti1.save(hdr, "res.hdr")