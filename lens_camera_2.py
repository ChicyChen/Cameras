# Version 2: Lens + One camera

import os

import numpy as np
import scipy.ndimage

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt


big = 300
#big = 1500
num = 5
#num = 5
big_grid_V = np.zeros((big*big,)) # |big*big| points
for i in range(big):
    for j in range(big):
        if (i//(big//num) + j//(big//num))%2:
           big_grid_V[i*big+j] = 0
        else:
           big_grid_V[i*big+j] = 1
big_grid_V = big_grid_V.astype(int)

side = 300
big_grid_P = np.zeros((big*big,3)) # side*side mm
for i in range(big):
    for j in range(big):
        big_grid_P[i*big+j] = np.array((i-(big//2),j-(big//2),0))
big_grid_P = big_grid_P.T * side / big


def generate_rotationM(axis, theta):
    """
    axis = "x" / "y" / "z"
    theta is in the form pi/2
    effect is to rotate WC to CC clockwise
    """
    if (axis == "x"):
        return np.array(((1,0,0),(0,np.cos(theta),-np.sin(theta)),(0,np.sin(theta),np.cos(theta))))
    if (axis == "y"):
        return np.array(((np.cos(theta),0,np.sin(theta)),(0,1,0),(-np.sin(theta),0,np.cos(theta))))
    if (axis == "z"):
        return np.array(((np.cos(theta),-np.sin(theta),0),(np.sin(theta),np.cos(theta),0),(0,0,1)))
    print("Wrong choice of axis, please choose x/y/z.\n")



def generate_extrinsic_matrix(Rx, Ry, Rz, t):
    """
    Func:
    Generate the extrinsic matrix for a camera.
    Representing the change from world coordinates to the camera coordinates
    for an object.

    Args:
    Rx: First rotate around the x-axis, (3,3)
    Ry: Second rotate around the y-axis, (3,3)
    Rz: Third rotate around the z-axis, (3,3)
    t: Translation of axis, (3,)

    Return:
    The extrinsic matrix
    """
    R = Rz @ Ry @ Rx
    extrinsic = np.hstack((R,t.reshape(-1,1)))
    return extrinsic



def generate_intrinsic_matrix(focal_length, p_x, p_y, c_x, c_y, skew = 0):
    """
    Func:
    Generate the intrinsic matrix for a camera.

    Args:
    focal_length
    (p_x, p_y): size of the pixels in world units
    (c_x, c_y): optical center (the principal point), in pixels
    skew: skew coefficient, which is non-zero if the image axes are not perpendicular

    Return:
    The intrinsic matrix
    """
    intrinsicA = np.array(((focal_length, 0, 0),
                         (0, focal_length, 0), 
                         (0, 0, 1)))
    intrinsicB = np.array(((1/p_x, 0, c_x),
                         (0, 1/p_y, c_y), 
                         (0, 0, 1)))
    return intrinsicA, intrinsicB
    


def point_world_to_image(P, intrinsicA, intrinsicB, extrinsic, distortion):
    """
    Func:
    Take a 3D point in the world coordinate, translate it to a 2D point in the image 
    coordinate. Notice the final result is in pixels, but is not integers yet.

    Args:
    P: (X,Y,Z), (3,)
    intrinsic: matrix
    extrinsic: matrix
    distortion: (k1,k2,k3,...)

    Return:
    The 2D point (u,v) in an image, (2,)
    """
    
    P = np.append(P, 1).reshape(-1,1)
    p = intrinsicA @ extrinsic @ P
    z = p[2]
    z[z==0] = 1e-10
    p = p / z
    r2 = p[0,0]**2 + p[1,0]**2
    p = (intrinsicB @ p)[0:2,:]
    
    u,v = intrinsicB[0,2], intrinsicB[1,2]
    xo = p[0]-u
    yo = p[1]-v
    
    n = distortion.shape[0]
    for i in range(n):
        p[0] = p[0] + distortion[i]*(r2**(i+1))*xo
        p[1] = p[1] + distortion[i]*(r2**(i+1))*yo
    return p.reshape(-1,)


def position(P, intrinsicA, intrinsicB, extrinsic, distortion):
    """
    Func:
    Take a matrix of the positions of a 3D plane in the world coordinate, translate it to a 
    matrix of positions of 2D point in the image coordinate. Notice the final result is in 
    pixels, but is not integers yet.

    Args:
    P: (3,-1), with the last 3 entries being x,y,z
    intrinsic: matrix
    extrinsic: matrix

    Return:
    The matrix of the 2D points in an image, (2,-1), i.e., ((u',v'),-1)
    """
    N = P.shape[1]
    result = np.zeros((N,2))
    for i in range(N):
        p = P[:,i]
        pn = point_world_to_image(p, intrinsicA, intrinsicB, extrinsic, distortion)
        result[i] = pn
    return result.T
    

def graph(P,V,im_size):
    """
    Func:
    Take a matrix of the positions of 2D points in the image coordinate, the value of each pixel,
    the size of the image, and the original z of each pixel,
    return the resulted image, where if two points have the same position, record the one with
    smaller z.

    Args:

    Return: img
    """
    mh,mw = im_size
    N = P.shape[1]
    img = np.ones((mh,mw))
    img = img / 2
    mea = np.zeros((mh,mw))
    for i in range(N):
        x,y = P[:,i]
        if (0<=x<mw-1 and 0<=y<mh-1):
            img[y,x] = (mea[y,x]*img[y,x] + V[i]) / (mea[y,x] + 1)
            img[y+1,x] = (mea[y+1,x]*img[y+1,x] + V[i]) / (mea[y+1,x] + 1)
            img[y,x+1] = (mea[y,x+1]*img[y,x+1] + V[i]) / (mea[y,x+1] + 1)
            img[y+1,x+1] = (mea[y+1,x+1]*img[y+1,x+1] + V[i]) / (mea[y+1,x+1] + 1)
            mea[y,x] += 1
            mea[y+1,x] += 1
            mea[y,x+1] += 1
            mea[y+1,x+1] += 1
    return img



def main():

    #focal_length, p_x, p_y, c_x, c_y = 30,1,1,150,150 #5mm,?,?
    focal_length, p_x, p_y, c_x, c_y = 5e-3,1e-5,1e-5,300,200 #5e3um,10um,10um
    #skew = focal_length / p_x * np.tan(np.pi/36)
    skew = 0
    intrinsicA, intrinsicB = generate_intrinsic_matrix(focal_length, p_x, p_y, c_x, c_y, skew)
    #Rx, Ry, Rz, t = np.eye(3),np.eye(3),np.eye(3),np.array((0,0,-600))
    #Rx, Ry, Rz, t = generate_rotationM("x",np.pi/12),np.eye(3),np.eye(3),np.array((0,0,-600))
    #Rx, Ry, Rz, t = np.eye(3),generate_rotationM("y",np.pi/12),np.eye(3),np.array((0,0,-600))
    #Rx, Ry, Rz, t = np.eye(3),np.eye(3),generate_rotationM("z",np.pi/12),np.array((0,0,-600))
    distance = 500 # distance mm far away from camera
    Rx, Ry, Rz, t = generate_rotationM("x",np.pi/18),generate_rotationM("y",np.pi/18),generate_rotationM("z",np.pi/18),np.array((0,0,-distance))
    extrinsic = generate_extrinsic_matrix(Rx, Ry, Rz, t)
    distortion = np.array((20000,0))
    #distortion = np.zeros(2)

    p = position(big_grid_P, intrinsicA, intrinsicB, extrinsic, distortion)
    p = p.astype(int)
    img = graph(p,big_grid_V,(400,600)) #400mm * 600mm photo
    plt.imsave("v2_grid_lens_small_resolution_2.png",img,cmap="gray")
    plt.imshow(img,cmap="gray")
    plt.show()
    
    return 0



if __name__ == '__main__':
    main()