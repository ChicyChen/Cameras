# Version 1: Pinhole + One camera

import os

import numpy as np
import scipy.ndimage

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

big = 1000
#big = 200
num = 5
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
    intrinsic = np.array(((focal_length / p_x, skew, c_x),
                         (0, focal_length / p_y, c_y), 
                         (0, 0, 1)))
    return intrinsic


def point_world_to_image(P, intrinsic, extrinsic):
    """
    Func:
    Take a 3D point in the world coordinate, translate it to a 2D point in the image 
    coordinate. Notice the final result is in pixels, but is not integers yet.

    Args:
    P: (X,Y,Z), (3,)
    intrinsic: matrix
    extrinsic: matrix

    Return:
    The 2D point (u,v) in an image, (2,)
    """
    
    P = np.append(P, 1).reshape(-1,1)
    p = intrinsic @ extrinsic @ P
    z = p[2]
    z[z==0] = 1e-10
    p = np.array((p[0],p[1])) / z
    return p.reshape(-1,)


def position(P, intrinsic, extrinsic):
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
    #origin_z = P[2]
    points = np.vstack((P,np.ones(P.shape[1])))
    result = intrinsic @ extrinsic @ points
    result = result / result[2].reshape(1,-1)
    result = result[0:2,:]
    return result
    

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
    #img = np.zeros((mh,mw))
    mea = np.zeros((mh,mw))
    for i in range(N):
        x,y = P[:,i]
        """
        if (0<=x<mw-1 and 0<=y<mh-1):
            img[y,x] = (mea[y,x]*img[y,x] + V[i]) / (mea[y,x] + 1)
            img[y+1,x] = (mea[y+1,x]*img[y+1,x] + V[i]) / (mea[y+1,x] + 1)
            img[y,x+1] = (mea[y,x+1]*img[y,x+1] + V[i]) / (mea[y,x+1] + 1)
            img[y+1,x+1] = (mea[y+1,x+1]*img[y+1,x+1] + V[i]) / (mea[y+1,x+1] + 1)
            mea[y,x] += 1
            mea[y+1,x] += 1
            mea[y,x+1] += 1
            mea[y+1,x+1] += 1
        """
        if (0<=x<mw and 0<=y<mh):
            img[y,x] = (mea[y,x]*img[y,x] + V[i]) / (mea[y,x] + 1)
            mea[y,x] += 1
    return img


def group_graphs(num_graph = 5):
    focal_length, p_x, p_y, c_x, c_y = 5e-3,1e-5,1e-5,300,200 #5e3um,10um,10um
    skew = 0
    intrinsic = generate_intrinsic_matrix(focal_length, p_x, p_y, c_x, c_y, skew)
    rotation_mats = []
    extrinsic_mats = []
    distance = 500 # distance mm far away from camera
    t = np.array((0,0,-distance))
    for i in range(num_graph):
        Rx, Ry, Rz = generate_rotationM("x",i*np.pi/36),generate_rotationM("y",i*np.pi/36),generate_rotationM("z",i*np.pi/36)
        extrinsic = generate_extrinsic_matrix(Rx, Ry, Rz, t)
        rotation_mats.append((Rx,Ry,Rz))
        extrinsic_mats.append(extrinsic)
        p = position(big_grid_P, intrinsic, extrinsic)
        p = p.astype(int)
        img = graph(p,big_grid_V,(400,600)) #400mm * 600mm photo
        plt.imsave(f"group_{i}.jpg",img,cmap="gray")
    return intrinsic, extrinsic_mats

def main():
    
    #focal_length, p_x, p_y, c_x, c_y = 30,1,1,150,150 #5mm,?,?
    focal_length, p_x, p_y, c_x, c_y = 5e-3,1e-5,1e-5,300,200 #5e3um,10um,10um
    #skew = focal_length / p_x * np.tan(np.pi/36)
    skew = 0
    intrinsic = generate_intrinsic_matrix(focal_length, p_x, p_y, c_x, c_y, skew)
    #Rx, Ry, Rz, t = np.eye(3),np.eye(3),np.eye(3),np.array((0,0,-600))
    #Rx, Ry, Rz, t = generate_rotationM("x",np.pi/12),np.eye(3),np.eye(3),np.array((0,0,-600))
    #Rx, Ry, Rz, t = np.eye(3),generate_rotationM("y",np.pi/12),np.eye(3),np.array((0,0,-600))
    #Rx, Ry, Rz, t = np.eye(3),np.eye(3),generate_rotationM("z",np.pi/12),np.array((0,0,-600))
    distance = 500 # distance mm far away from camera
    Rx, Ry, Rz, t = generate_rotationM("x",np.pi/18),generate_rotationM("y",np.pi/18),generate_rotationM("z",np.pi/18),np.array((0,0,-distance))
    extrinsic = generate_extrinsic_matrix(Rx, Ry, Rz, t)
    p = position(big_grid_P, intrinsic, extrinsic)
    p = p.astype(int)
    img = graph(p,big_grid_V,(400,600)) #400mm * 600mm photo
    plt.imsave("v2_grid_high_resolution_withoutfilter.png",img,cmap="gray")
    plt.imshow(img,cmap="gray")
    plt.show()
    
    #intrinsic, extrinsic_mats = group_graphs()
    return 0



if __name__ == '__main__':
    main()