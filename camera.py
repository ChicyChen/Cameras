from scipy import linalg
import numpy as np

class Camera(object):
  """ Class for representing pin-hole cameras. """

  def __init__(self,P):
    """ Initialize P = K[R|t] camera model. """
    self.P = P
    self.K = None # calibration matrix
    self.R = None # rotation
    self.t = None # translation
    self.c = None # camera center


  def project(self,X):
    """  Project points in X (4*n array) and normalize coordinates. """

    x = np.dot(self.P,X)
    for i in range(3):
      x[i] /= x[2]
    return x

def rotation_matrix(a):
    """  
    Creates a 3D rotation matrix for rotation
    around the axis of the vector a. 
    """
    R = np.eye(4)
    R[:3,:3] = linalg.expm([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return R