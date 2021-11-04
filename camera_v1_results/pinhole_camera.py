# Version 1: Pinhole + One camera

import os

import numpy as np
import scipy.ndimage

from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

grid_color = np.zeros((5,5,3))
for i in range(5):
    for j in range(5):
        if (i+j)%2:
           grid_color[i,j] = np.array((0,255,0))
        else:
           grid_color[i,j] = np.array((255,255,255))
grid_color = grid_color.astype(int)

grid_pos = np.zeros((5,5,3))
for i in range(5):
    for j in range(5):
        grid_pos[i,j] = np.array((i-2,j-2,0))

big = 1500
num = 3
big_grid_V = np.zeros((big,big,3)) #0.2m*0.2m
for i in range(big):
    for j in range(big):
        if (i//(big//num) + j//(big//num))%2:
           big_grid_V[i,j] = np.array((255,0,0))
        else:
           big_grid_V[i,j] = np.array((255,255,255))
big_grid_V = big_grid_V.astype(int)

big_grid_P = np.zeros((big,big,3))
for i in range(big):
    for j in range(big):
        big_grid_P[i,j] = np.array((i-(big//2),j-(big//2),0))


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



def read_parameters():
    return 0


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
    P: (H,W,3), with the last 3 entries being x,y,z
    intrinsic: matrix
    extrinsic: matrix

    Return:
    The matrix of the 2D points in an image, (H,W,(u',v'))
    """
    H,W = P.shape[0],P.shape[1]
    origin_z = np.zeros((H,W))
    result = np.zeros((H,W,2))
    for i in range(H):
        for j in range(W):
            p = P[i,j]
            pn = point_world_to_image(p, intrinsic, extrinsic)
            result[i,j] = pn
            """
            pc = extrinsic @ np.append(p, 1).reshape(-1,1)
            origin_z[i,j] = pc[2]
            """
    return result, origin_z
    

def graph(P,V,im_size,origin_z):
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
    H,W = P.shape[0],P.shape[1]
    if(len(V.shape)==3):
        img = np.zeros((mh,mw,3))
    else:
        img = np.zeros((mh,mw))
    img_z = np.zeros((mh,mw))
    for i in range(H):
        for j in range(W):
            x,y = P[i,j]
            #print(P[i,j])
            if (0<=x<mw and 0<=y<mh):
                if (img_z[y,x] == 0  or img_z[y,x]>origin_z[i,j]):
                    img[y,x] = V[i,j]
                    img_z[y,x] = origin_z[i,j]
            #continue
    return img


"""
plt can't use 3D array to construct 3D image directly.
"""
def graph3D(P,V,im_size):
    """
    Func:
    Take a matrix of the positions of 3D points in the world coordinate, and the value of each pixel,
    return the resulted 3D image

    Args:

    Return:
    """
    mh,mw,mz = im_size
    H,W = P.shape[0],P.shape[1]
    img = np.zeros((mh,mw,mz))
    for i in range(H):
        for j in range(W):
            x,y,z = P[i,j]
            #print(P[i,j])
            if (0<=x<mw and 0<=y<mh and 0<=z<mz):
                img[y,x,z] = V[i,j]
            #continue
    return img


def test_ptop():
    """
    Test function for point_world_to_image
    """
    focal_length, p_x, p_y, c_x, c_y = 2,1,1,3,3
    intrinsic = generate_intrinsic_matrix(focal_length, p_x, p_y, c_x, c_y, skew = 0)
    Rx, Ry, Rz, t = generate_rotationM("x",np.pi / 4),np.eye(3),np.eye(3),np.array((0,0,-2))
    extrinsic = generate_extrinsic_matrix(Rx, Ry, Rz, t)
    p = point_world_to_image(np.array((-1,2,0)),intrinsic, extrinsic)
    print(p)
    pass

def test_graph():
    """
    Test function for graph
    """
    focal_length, p_x, p_y, c_x, c_y = 50,1,1,250,250
    #focal_length, p_x, p_y, c_x, c_y = 20,1,1,100,100
    intrinsic = generate_intrinsic_matrix(focal_length, p_x, p_y, c_x, c_y, skew = 0)

    #Rx, Ry, Rz, t = np.eye(3),np.eye(3),np.eye(3),np.array((0,0,-50))
    Rx, Ry, Rz, t = generate_rotationM("x",np.pi/4),np.eye(3),np.eye(3),np.array((0,0,-50))
    #Rx, Ry, Rz, t = np.eye(3),generate_rotationM("y",np.pi/4),np.eye(3),np.array((0,0,-50))
    #Rx, Ry, Rz, t = np.eye(3),np.eye(3),generate_rotationM("z",np.pi/6),np.array((0,0,-50))

    #Rx, Ry, Rz, t = np.eye(3),np.eye(3),np.eye(3),np.array((0,0,-2))
    #Rx, Ry, Rz, t = generate_rotationM("x",np.pi/4),np.eye(3),np.eye(3),np.array((0,0,-2))
    #Rx, Ry, Rz, t = np.eye(3),generate_rotationM("y",np.pi/4),np.eye(3),np.array((0,0,-2))
    #Rx, Ry, Rz, t = np.eye(3),np.eye(3),generate_rotationM("z",np.pi/6),np.array((0,0,-2))
    extrinsic = generate_extrinsic_matrix(Rx, Ry, Rz, t)

    P = np.random.randint(100, size=(100, 100, 3))
    V = np.random.randint(255, size=(100, 100))
    #print(p)
    #print(V)

    #p,origin_z = position(grid_pos, intrinsic, extrinsic)
    p,origin_z = position(P, intrinsic, extrinsic)
    p = p.astype(int)

    img = graph(p,V.astype(int),(500,500),origin_z)
    plt.imshow(img)
    plt.show()

    #img = graph(p,grid_color.astype(int),(200,200),origin_z)
    #plt.imsave("grid_origin.png",img.astype(np.uint8))
    #plt.imsave("grid_rx.png",img.astype(np.uint8))
    #plt.imsave("grid_ry.png",img.astype(np.uint8))
    #plt.imsave("grid_rz.png",img.astype(np.uint8))
    #plt.imshow(img)
    #plt.show()
    
    pass


def main():
    """
    V = plt.imread("colors.png")
    print(V)
    H,W = V.shape[0],V.shape[1]

    focal_length, p_x, p_y, c_x, c_y = 5,1,1,W*2,H*2 #5mm,?,?
    intrinsic = generate_intrinsic_matrix(focal_length, p_x, p_y, c_x, c_y, skew = 0)

    Rx, Ry, Rz, t = np.eye(3),np.eye(3),np.eye(3),np.array((0,0,-focal_length))
    #Rx, Ry, Rz, t = generate_rotationM("x",np.pi/12),np.eye(3),np.eye(3),np.array((0,0,-focal_length // 2))
    #Rx, Ry, Rz, t = np.eye(3),generate_rotationM("y",np.pi/12),np.eye(3),np.array((0,0,-focal_length // 2))
    #Rx, Ry, Rz, t = np.eye(3),np.eye(3),generate_rotationM("z",np.pi/12),np.array((0,0,-focal_length // 2))
    #Rx, Ry, Rz, t = generate_rotationM("x",np.pi/12),generate_rotationM("y",np.pi/12),np.eye(3),np.array((0,0,-focal_length // 2))
    extrinsic = generate_extrinsic_matrix(Rx, Ry, Rz, t)

    P = np.zeros((H,W,3))
    for i in range(H):
        P[i,:,0] = np.arange(W) - W//2
        P[i,:,1] = i - H//2

    p,origin_z = position(P, intrinsic, extrinsic)
    p = p.astype(int)
    print(p)

    img = graph(p,V,(W*4,H*4),origin_z)
    plt.imshow(img)
    plt.show()
    """

    focal_length, p_x, p_y, c_x, c_y = 30,1,1,150,150 #5mm,?,?
    skew = focal_length / p_x * np.tan(np.pi/36)
    intrinsic = generate_intrinsic_matrix(focal_length, p_x, p_y, c_x, c_y, skew)
    #Rx, Ry, Rz, t = np.eye(3),np.eye(3),np.eye(3),np.array((0,0,-600))
    #Rx, Ry, Rz, t = generate_rotationM("x",np.pi/12),np.eye(3),np.eye(3),np.array((0,0,-600))
    #Rx, Ry, Rz, t = np.eye(3),generate_rotationM("y",np.pi/12),np.eye(3),np.array((0,0,-600))
    #Rx, Ry, Rz, t = np.eye(3),np.eye(3),generate_rotationM("z",np.pi/12),np.array((0,0,-600))
    Rx, Ry, Rz, t = generate_rotationM("x",np.pi/36),generate_rotationM("y",np.pi/36),generate_rotationM("z",np.pi/36),np.array((0,0,-600))
    extrinsic = generate_extrinsic_matrix(Rx, Ry, Rz, t)
    p,origin_z = position(big_grid_P, intrinsic, extrinsic)
    p = p.astype(int)
    #print(p.dtype)
    img = graph(p,big_grid_V.astype(np.uint8),(300,300),origin_z) #200mm * 200mm photo
    plt.imsave("big_grid_skew.png",img.astype(np.uint8))
    plt.imshow(img)
    plt.show()

    
    return 0





if __name__ == '__main__':
    main()