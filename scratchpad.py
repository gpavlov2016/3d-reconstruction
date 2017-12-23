import numpy as np
from sklearn import mixture
from sklearn.mixture import GMM
#from sklearn.mixture import GaussianMixture
import numpy as np
import math
import cv2
from matplotlib import pyplot as plt


def compute_fundamental(x1,x2):
    """ Computes the fundamental matrix from corresponding points
    (x1,x2 3*n arrays) using the normalized 8 point algorithm.
    each row is constructed as
    [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """
    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")
    # build matrix for equations
    A = zeros((n,9))
    for i in range(n):
        A[i] = [x1[0,i]*x2[0,i], x1[0,i]*x2[1,i], x1[0,i]*x2[2,i],
        x1[1,i]*x2[0,i], x1[1,i]*x2[1,i], x1[1,i]*x2[2,i],
        x1[2,i]*x2[0,i], x1[2,i]*x2[1,i], x1[2,i]*x2[2,i] ]
    # compute linear least square solution
    U,S,V = linalg.svd(A)
    F = V[-1].reshape(3,3)
    # constrain F
    # make rank 2 by zeroing out last singular value
    U,S,V = linalg.svd(F)
    S[2] = 0
    F = dot(U,dot(diag(S),V))
    return F



from numpy import *
import camera
from matplotlib.pyplot import *

# load some images
im1 = cv2.imread('img3/001.jpg')
im2 = cv2.imread('img3/002.jpg')
#im1 = cv2.imread('img2/100_7101.jpg')
#im2 = cv2.imread('img2/100_7106.jpg')

# load 2D points for each view to a list
points2D = [np.loadtxt('2D/00'+str(i+1)+'.corners').T for i in range(3)]
# load 3D points
points3D = loadtxt('3D/p3d').T
# load correspondences
corr = np.genfromtxt('2D/nview-corners',dtype='int')
# load cameras to a list of Camera objects
P = [camera.Camera(np.loadtxt('2D/00'+str(i+1)+'.P')) for i in range(3)]


# make 3D points homogeneous and project
X = vstack((points3D, ones(points3D.shape[1])))
x = P[0].project(X)
# plotting the points in view 1
figure()
imshow(im1)
plot(points2D[0][0], points2D[0][1],' * ')
axis('off')
figure()
imshow(im1)
plot(x[0], x[1],'r.')
axis('off')

from mpl_toolkits.mplot3d import axes3d
fig = figure()
ax = fig.gca(projection="3d")
# generate 3D sample data
X,Y,Z = axes3d.get_test_data(0.25)
# plot the points in 3D
ax.plot(X.flatten(),Y.flatten(),Z.flatten(),'o')


# plotting 3D points
from mpl_toolkits.mplot3d import axes3d
fig = figure()
ax = fig.gca(projection='3d')
ax.plot(points3D[0],points3D[1],points3D[2],'k.')



import sfm
# index for points in first two views
ndx = (corr[:,0]>=0) & (corr[:,1]>=0)
# get coordinates and make homogeneous
x1 = points2D[0][:,corr[ndx,0]]
x1 = vstack( (x1,ones(x1.shape[1])) )
x2 = points2D[1][:,corr[ndx,1]]
x2 = vstack( (x2,ones(x2.shape[1])) )
print(x1.shape)
# compute F
F = sfm.compute_fundamental(x1,x2)
# compute the epipole
e = sfm.compute_epipole(F)
# plotting
figure()
imshow(im1)
# plot each line individually, this gives nice colors
for i in range(5):
    sfm.plot_epipolar_line(im1,F,x2[:,i],e,False)
axis('off')
figure()
imshow(im2)
# plot each point individually, this gives same colors as the lines
for i in range(5):
    plot(x2[0,i],x2[1,i],'o')
axis('off')
show()











exit(0)







def from_video():
    import pylab
    import imageio
    filename = 'toothpick.mp4'
    vid = imageio.get_reader(filename,  'ffmpeg')
    images = []
    for i in range(210): #350
      image = vid.get_data(i)
      images.append(image)

    imgL = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)#cv2.imread('tsukuba_l.png',0)
    imgR = cv2.cvtColor(images[150], cv2.COLOR_BGR2GRAY)#cv2.imread('tsukuba_r.png',0)

    plt.figure()
    plt.imshow(imgL, 'gray')
    plt.figure()
    plt.imshow(imgR, 'gray')
    plt.show()

    cv2.imwrite('im1.png', imgL)
    cv2.imwrite('im2.png', imgR)


from_video()

exit(0)


# Fit
mu, sigma = 120, 10
mu1, sigma1 = 150, 10
X = np.random.normal(mu, sigma, (100,1))
X = np.concatenate((np.random.normal(mu, sigma, (10,1)),
                    np.random.normal(mu1, sigma1, (50,1))))
clf = mixture.GMM(n_components=2)
clf = mixture.GMM(n_components=2)
print(clf.fit(X))
print(clf.means_)
print(clf.covars_)
# print("cov:", clf.covariances_, "std 1:", math.sqrt(clf.covariances_[0]), "std 2:", math.sqrt(clf.covariances_[1]))
# print(clf.score(X))

# Predict
X_test = np.array([range(100, 200)]).T
X_test = np.random.normal(mu, sigma, (100,1))
#X_test = np.array(120)
#pred = clf.predict_proba(X_test)
#print(pred)

lh = clf.score(X_test)
ll = np.exp(lh)
print(ll)

print(ll.mean())

print(ll.sum())

plt.plot(ll)
plt.show()
cv2.waitKey(1000)

exit(0)

np.random.seed(1)
g = mixture.GMM(n_components=2)
# Generate random observations with two modes centered on 0
# and 10 to use for training.
obs = np.concatenate((np.random.randn(100, 1),
                       10 + np.random.randn(300, 1)))
g.fit(obs)
GMM(n_components=2)
ret = np.round(g.weights_, 2)
print(ret)
#array([ 0.75,  0.25])
ret = np.round(g.means_, 2)
print(ret)
#array([[ 10.05],
#       [  0.06]])
ret = np.round(g.covars_, 2)
print(ret)
#array([[[ 1.02]],
#       [[ 0.96]]])
ret = g.predict([[0], [2], [9], [10]])
print(ret)

# array([1, 1, 0, 0])
ret= np.round(g.score([[0], [2], [9], [10]]), 2)
print(ret)

#array([-2.19, -4.58, -1.75, -1.21])
# Refit the model on new data (initial parameters remain the
# same), this time with an even split between the two modes.
g.fit(20 * [[0]] +  20 * [[10]])
GMM(n_components=2)
ret = np.round(g.weights_, 2)
print(ret)
plt.plot(ret)
plt.show()
#cv2.waitKey(1000)

#array([ 0.5,  0.5])