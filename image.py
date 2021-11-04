import camera
import numpy as np
from matplotlib import pyplot as plt

# load points
points = np.loadtxt('3D/house.p3d').T
points = np.vstack((points,np.ones(points.shape[1])))
print(points.shape)

# setup camera
P = np.hstack((np.eye(3),np.array([[0],[0],[-10]])))
cam = camera.Camera(P)
x = cam.project(points)

# plot projection
plt.figure()
plt.plot(x[0],x[1],'k.')
plt.show()

# create transformation
r = 0.05*np.random.rand(3)
rot = camera.rotation_matrix(r)

# rotate camera and project
plt.figure()
for t in range(20):
  cam.P = np.dot(cam.P,rot)
  x = cam.project(points)
  plt.plot(x[0],x[1],'k.')
plt.show()