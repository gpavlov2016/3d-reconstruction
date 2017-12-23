import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    rows = img1.shape[0]
    cols = img1.shape[1]
    #img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for line,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        #x0,y0 = map(int, [0, -r[2]/r[1] ])
        #x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        x0,y0 = map(int, [0, -line[2] / line[1] ])
        x1,y1 = map(int, [cols, -(line[2] + line[0] * cols) / line[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,3)
        #img1 = cv2.circle(img1,tuple(pt1[0]),30,color,-1)
        #img2 = cv2.circle(img2,tuple(pt2[0]),30,color,-1)
    return img1,img2


def dist_from_corner(pt, corner):
    return math.sqrt((pt[0]-corner[0])**2 + (pt[1]-corner[1])**2)


def testing(img1, img2, intrinsics, dist):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2

    img1 = cv2.resize(img1, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)

    sift = cv2.xfeatures2d.SIFT_create()
    cv2.bgsegm.createBackgroundSubtractorMOG()
    cv2.bgsegm.createBackgroundSubtractorGMG()
    # Initiate SIFT detector
    #sift = cv2.SIFT()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()


from math  import sin, cos

def rotate_image(img, alpha, beta, gamma, dx, dy, dz, f):
    #alpha = (alpha - 90.)*np.pi/180.;
    #beta = (beta - 90.)*np.pi/180.;
    #gamma = (gamma - 90.)*np.pi/180.;
    # get width and height for ease of use in matrices
    w = img.shape[1]
    h = img.shape[0]
    # Projection 2D -> 3D matrix
    A1 = np.array([
              [1, 0, -w/2],
              [0, 1, -h/2],
              [0, 0,    0],
              [0, 0,    1]])
    # Rotation matrices around the X, Y, and Z axis
    RX = np.array([
              [1,          0,           0, 0],
              [0, cos(alpha), -sin(alpha), 0],
              [0, sin(alpha),  cos(alpha), 0],
              [0,          0,           0, 1]])
    RY = np.array([
              [cos(beta), 0, -sin(beta), 0],
              [0, 1,          0, 0],
              [sin(beta), 0,  cos(beta), 0],
              [0, 0,          0, 1]])
    RZ = np.array([
              [cos(gamma), -sin(gamma), 0, 0],
              [sin(gamma),  cos(gamma), 0, 0],
              [0,          0,           1, 0],
              [0,          0,           0, 1]])
    # Composed rotation matrix with (RX, RY, RZ)
    R = RX.dot(RY).dot(RZ)
    # Translation matrix
    T = np.array([
             [1, 0, 0, dx],
             [0, 1, 0, dy],
             [0, 0, 1, dz],
             [0, 0, 0, 1]])
    # 3D -> 2D matrix
    A2 = np.array([
              [f, 0, w/2, 0],
              [0, f, h/2, 0],
              [0, 0,   1, 0]])
    # Final transformation matrix
    print('A2.dot(T):', A2.dot(T))
    trans = ((A2.dot(T)).dot(R)).dot(A1)
    #trans = np.eye(3,3)
    #trans[0,0] = 0
    # Apply matrix transformation
    print('trans: ', trans)
    ret_img = cv2.warpPerspective(img, trans, (w,h), cv2.INTER_LINEAR)
    plt.imshow(ret_img)
    plt.show()
    return ret_img


def validate_F(F, pts1, pts2):
    pt1 = np.array([[pts1[0][0][0]], [pts1[0][0][1]], [1]])
    pt2 = np.array([[pts2[0][0][0], pts2[0][0][1], 1]])
    print('val:', pt2.dot(F).dot(pt1))


def epi_scipy(img1, img2):
    import numpy as np
    from skimage import data
    from skimage.color import rgb2gray
    from skimage.feature import match_descriptors, ORB, plot_matches
    from skimage.measure import ransac
    from skimage.transform import FundamentalMatrixTransform
    import matplotlib.pyplot as plt

    np.random.seed(0)

    img_left, img_right, groundtruth_disp = data.stereo_motorcycle()
    img_left, img_right = map(rgb2gray, (img_left, img_right))

    # Find sparse feature correspondences between left and right image.

    descriptor_extractor = ORB()

    descriptor_extractor.detect_and_extract(img_left)
    keypoints_left = descriptor_extractor.keypoints
    descriptors_left = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(img_right)
    keypoints_right = descriptor_extractor.keypoints
    descriptors_right = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors_left, descriptors_right,
                                cross_check=True)

    # Estimate the epipolar geometry between the left and right image.

    model, inliers = ransac((keypoints_left[matches[:, 0]],
                             keypoints_right[matches[:, 1]]),
                            FundamentalMatrixTransform, min_samples=8,
                            residual_threshold=1, max_trials=5000)

    inlier_keypoints_left = keypoints_left[matches[inliers, 0]]
    inlier_keypoints_right = keypoints_right[matches[inliers, 1]]

    print("Number of matches:", matches.shape[0])
    print("Number of inliers:", inliers.sum())

    # Compare estimated sparse disparities to the dense ground-truth disparities.

    disp = inlier_keypoints_left[:, 1] - inlier_keypoints_right[:, 1]
    disp_coords = np.round(inlier_keypoints_left).astype(np.int64)
    disp_idxs = np.ravel_multi_index(disp_coords.T, groundtruth_disp.shape)
    disp_error = np.abs(groundtruth_disp.ravel()[disp_idxs] - disp)
    disp_error = disp_error[np.isfinite(disp_error)]

    # Visualize the results.

    fig, ax = plt.subplots(nrows=2, ncols=1)

    plt.gray()

    plot_matches(ax[0], img_left, img_right, keypoints_left, keypoints_right,
                 matches[inliers], only_matches=True)
    ax[0].axis("off")
    ax[0].set_title("Inlier correspondences")

    ax[1].hist(disp_error)
    ax[1].set_title("Histogram of disparity errors")

    plt.show()



def flann_matching(img1, img2):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, outImg=ret)
    plt.imshow(img3), plt.show()
    return good


from pylab import *
from numpy import *
from scipy import linalg


def compute_P(x, X):
    """    Compute camera matrix from pairs of
        2D-3D correspondences (in homog. coordinates). """

    n = x.shape[1]
    if X.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # create matrix for DLT solution
    M = zeros((3 * n, 12 + n))
    for i in range(n):
        M[3 * i, 0:4] = X[:, i]
        M[3 * i + 1, 4:8] = X[:, i]
        M[3 * i + 2, 8:12] = X[:, i]
        M[3 * i:3 * i + 3, i + 12] = -x[:, i]

    U, S, V = linalg.svd(M)

    return V[-1, :12].reshape((3, 4))


def triangulate_point(x1, x2, P1, P2):
    """ Point pair triangulation from
        least squares solution. """

    M = zeros((6, 6))
    M[:3, :4] = P1
    M[3:, :4] = P2
    M[:3, 4] = -x1
    M[3:, 5] = -x2

    U, S, V = linalg.svd(M)
    X = V[-1, :4]

    return X / X[3]


def triangulate(x1, x2, P1, P2):
    """    Two-view triangulation of points in
        x1,x2 (3*n homog. coordinates). """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    X = [triangulate_point(x1[:, i], x2[:, i], P1, P2) for i in range(n)]
    return array(X).T


def compute_fundamental(x1, x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # build matrix for equations
    A = zeros((n, 9))
    for i in range(n):
        A[i] = [x1[0, i] * x2[0, i], x1[0, i] * x2[1, i], x1[0, i] * x2[2, i],
                x1[1, i] * x2[0, i], x1[1, i] * x2[1, i], x1[1, i] * x2[2, i],
                x1[2, i] * x2[0, i], x1[2, i] * x2[1, i], x1[2, i] * x2[2, i]]

    # compute linear least square solution
    U, S, V = linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = linalg.svd(F)
    S[2] = 0
    F = dot(U, dot(diag(S), V))

    return F / F[2, 2]


def compute_epipole(F):
    """ Computes the (right) epipole from a
        fundamental matrix F.
        (Use with F.T for left epipole.) """

    # return null space of F (Fx=0)
    U, S, V = linalg.svd(F)
    e = V[-1]
    return e / e[2]


def plot_epipolar_line(im, F, x, epipole=None, show_epipole=True):
    """ Plot the epipole and epipolar line F*x=0
        in an image. F is the fundamental matrix
        and x a point in the other image."""

    m, n = im.shape[:2]
    line = dot(F, x)

    # epipolar line parameter and values
    t = linspace(0, n, 100)
    lt = array([(line[2] + line[0] * tt) / (-line[1]) for tt in t])

    # take only line points inside the image
    ndx = (lt >= 0) & (lt < m)
    plot(t[ndx], lt[ndx], linewidth=2)

    if show_epipole:
        if epipole is None:
            epipole = compute_epipole(F)
        plot(epipole[0] / epipole[2], epipole[1] / epipole[2], 'r*')


def skew(a):
    """ Skew matrix A such that a x v = Av for any v. """

    return array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])


def compute_P_from_fundamental(F):
    """    Computes the second camera matrix (assuming P1 = [I 0])
        from a fundamental matrix. """

    e = compute_epipole(F.T)  # left epipole
    Te = skew(e)
    return vstack((dot(Te, F.T).T, e)).T


def compute_P_from_essential(E):
    """    Computes the second camera matrix (assuming P1 = [I 0])
        from an essential matrix. Output is a list of four
        possible camera matrices. """

    # make sure E is rank 2
    U, S, V = svd(E)
    if det(dot(U, V)) < 0:
        V = -V
    E = dot(U, dot(diag([1, 1, 0]), V))

    # create matrices (Hartley p 258)
    Z = skew([0, 0, -1])
    W = array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # return all four solutions
    P2 = [vstack((dot(U, dot(W, V)).T, U[:, 2])).T,
          vstack((dot(U, dot(W, V)).T, -U[:, 2])).T,
          vstack((dot(U, dot(W.T, V)).T, U[:, 2])).T,
          vstack((dot(U, dot(W.T, V)).T, -U[:, 2])).T]

    return P2


def compute_fundamental_normalized(x1, x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the normalized 8 point algorithm. """

    n = x1.shape[1]
    if x2.shape[1] != n:
        raise ValueError("Number of points don't match.")

    # normalize image coordinates
    x1 = x1 / x1[2]
    mean_1 = mean(x1[:2], axis=1)
    S1 = sqrt(2) / std(x1[:2])
    T1 = array([[S1, 0, -S1 * mean_1[0]], [0, S1, -S1 * mean_1[1]], [0, 0, 1]])
    x1 = dot(T1, x1)

    x2 = x2 / x2[2]
    mean_2 = mean(x2[:2], axis=1)
    S2 = sqrt(2) / std(x2[:2])
    T2 = array([[S2, 0, -S2 * mean_2[0]], [0, S2, -S2 * mean_2[1]], [0, 0, 1]])
    x2 = dot(T2, x2)

    # compute F with the normalized coordinates
    F = compute_fundamental(x1, x2)

    # reverse normalization
    F = dot(T1.T, dot(F, T2))

    return F / F[2, 2]


class RansacModel(object):
    """ Class for fundmental matrix fit with ransac.py from
        http://www.scipy.org/Cookbook/RANSAC"""

    def __init__(self, debug=False):
        self.debug = debug

    def fit(self, data):
        """ Estimate fundamental matrix using eight
            selected correspondences. """

        # transpose and split data into the two point sets
        data = data.T
        x1 = data[:3, :8]
        x2 = data[3:, :8]

        # estimate fundamental matrix and return
        F = compute_fundamental_normalized(x1, x2)
        return F

    def get_error(self, data, F):
        """ Compute x^T F x for all correspondences,
            return error for each transformed point. """

        # transpose and split data into the two point
        data = data.T
        x1 = data[:3]
        x2 = data[3:]

        # Sampson distance as error measure
        Fx1 = dot(F, x1)
        Fx2 = dot(F, x2)
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        err = (diag(dot(x1.T, dot(F, x2)))) ** 2 / denom

        # return error per point
        return err


def F_from_ransac(x1, x2, model, maxiter=5000, match_theshold=1e-6):
    """ Robust estimation of a fundamental matrix F from point
        correspondences using RANSAC (ransac.py from
        http://www.scipy.org/Cookbook/RANSAC).

        input: x1,x2 (3*n arrays) points in hom. coordinates. """

    import ransac

    data = vstack((x1, x2))

    # compute F and return with inlier index
    F, ransac_data = ransac.ransac(data.T, model, 8, maxiter, match_theshold, 20, return_all=True)
    return F, ransac_data['inliers']


def compute_epipole(F):
    """ Computes the (right) epipole from a
    fundamental matrix F.
    (Use with F.T for left epipole.) """
    # return null space of F (Fx=0)
    U,S,V = linalg.svd(F)
    e = V[-1]
    return e/e[2]


def compute_fundamental(pts1,pts2):
    """ Computes the fundamental matrix from corresponding points
    (x1,x2 3*n arrays) using the normalized 8 point algorithm.
    each row is constructed as
    [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """
    x1 = pts1[:,0,:]
    x2 = pts2[:,0,:]
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


def match_features(img1, img2, intrinsics, dist):

    w = img1.shape[1]
    h = img1.shape[0]
    corners = [[0,0], [0,w-1], [h-1,w-1], [h-1, 0]]
    # Initiate ORB detector
    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE, nfeatures=10000)
    # find the keypoints with ORB
    kp1 = orb.detect(img1, None)
    kp2 = orb.detect(img2, None)
    # compute the descriptors with ORB
    kp1, des1 = orb.compute(img1, kp1)
    #print('kp1: ', kp1[0].pt)
    #print('des1:', des1)
    kp2, des2 = orb.compute(img2, kp2)
    # draw only keypoints location,not size and orientation
    vis = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=2)
    #plt.imshow(vis), plt.show()

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des1, des2)
    #matches = flann_matching(img1, img2)
    #matches = np.array(matches)[:,0]
    #print(matches.shape)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    #print('dist:', [x.distance for x in matches])

    img3 = np.zeros_like(img2)
    # Draw first 10 matches.
    n_matches = 100
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:n_matches], img3, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img3)
    plt.show()

    focal = intrinsics[0,0]
    ppx = intrinsics[0,2]
    ppy = intrinsics[1,2]
    alignedLeft = {'pt': [], 'des': []}
    alignedRight = {'pt': [], 'des': []}
    leftBackReference = []
    rightBackReference = []
    #Arrange matching points in aligned arrays:
    for i, match in enumerate(matches[:n_matches]):
        qid = match.queryIdx
        tid = match.trainIdx
        #print('qid, tid:', qid, tid)
        alignedLeft['pt'].append(kp1[qid].pt)
        alignedLeft['des'].append(des1[qid,:])
        alignedRight['pt'].append(kp2[tid].pt)
        alignedRight['des'].append(des2[tid,:])
    pts1 = np.array(alignedLeft['pt']).reshape(n_matches, 1, -1).astype(np.float32)
    pts2 = np.array(alignedRight['pt']).reshape(n_matches, 1, -1).astype(np.float32)


    for i in range(10):
        print(dist_from_corner(pts1[i,0,:], pts2[i,0,:]))

    '''
    M, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    print('M:', M)

    d = 400
    pts = np.float32([[d, d], [d, h - 1 - d], [w - 1 - d, h - 1 - d], [w - 1 - d, d]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    plt.imshow(img2), plt.show()

    pts = np.float32([[d, d], [d, h - 1 - d], [w - 1 - d, h - 1 - d], [w - 1 - d, d],
                      [d/2, d/2], [d/2, h - 1 - d/2], [w - 1 - d/2, h - 1 - d/2], [w - 1 - d/2, d/2]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    print(pts1)
    print(dst)
    pts1 = pts
    pts2 = dst
    F, mask = cv2.findFundamentalMat(pts, dst, cv2.FM_RANSAC)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    print('F: ', F)
    validate_F(F, pts1, pts2)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts1, 1, F)
    lines1 = lines1.reshape(-1, 3)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts2, 2, F)
    lines2 = lines2.reshape(-1, 3)
    print('lines1: ', lines1)
    print('lines2: ', lines2)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    #plt.subplot(223), plt.imshow(img3)
    #plt.subplot(224), plt.imshow(img4)
    plt.show()

    exit(0)
    '''

    #plt.subplot(221), plt.scatter(pts1[:,0,0], pts1[:,0,1]), plt.title('pts1'),  plt.xlim(0, w), plt.ylim(0, h), plt.gca().invert_yaxis()
    #plt.subplot(222), plt.scatter(pts2[:,0,0], pts2[:,0,1]), plt.title('pts2'),  plt.xlim(0, w), plt.ylim(0, h), plt.gca().invert_yaxis()
    #plt.subplot(223), plt.imshow(img1, 'gray'), plt.title('img1')
    #plt.subplot(224), plt.imshow(img2, 'gray'), plt.title('img2')
    #plt.show()

    '''
    pts_uv1 = cv2.undistortPoints(pts1, intrinsics, distCoeffs=None, P=intrinsics)
    pts_uv2 = cv2.undistortPoints(pts2, intrinsics, distCoeffs=None, P=intrinsics)

    for pt in pts_uv1[:,0,:]:
        cv2.circle(img1, (pt[0], pt[1]), 10, (255), -1)
    for pt in pts_uv2[:,0,:]:
        cv2.circle(img2, (pt[0], pt[1]), 10, (255), -1)


    plt.subplot(221), plt.scatter(pts_uv1[:,0,0], pts_uv1[:,0,1]), plt.title('pts1'),  plt.xlim(0, w), plt.ylim(0, h), plt.gca().invert_yaxis()
    plt.subplot(222), plt.scatter(pts_uv2[:,0,0], pts_uv2[:,0,1]), plt.title('pts2'),  plt.xlim(0, w), plt.ylim(0, h), plt.gca().invert_yaxis()
    plt.subplot(223), plt.imshow(img1, 'gray'), plt.title('img1')
    plt.subplot(224), plt.imshow(img2, 'gray'), plt.title('img2')
    #plt.show()

    pts_uv1_t = np.reshape(pts_uv1, (1,-1,2))[0,0:4,:]
    pts_uv2_t = np.reshape(pts_uv2, (1,-1,2))[0,0:4,:]
    print('pts_uv1_t:', pts_uv1_t, pts_uv1_t.dtype)
    print('pts_uv2_t:', pts_uv2_t, pts_uv2_t.dtype)

    pts_uv1_t = np.reshape(pts1, (1,-1,2))[0,0:4,:]
    pts_uv2_t = np.reshape(pts2, (1,-1,2))[0,0:4,:]

    M = cv2.getPerspectiveTransform(pts_uv1_t, pts_uv2_t)

    print('M: ', M)

    imt = cv2.imread('chess.jpg')

    #M = np.eye(3)
    dst = cv2.warpPerspective(imt, M, (imt.shape[1], imt.shape[0]))
    #print(dst)

    plt.subplot(121), plt.imshow(img1, 'gray'), plt.title('Input')
    plt.subplot(122), plt.imshow(dst), plt.title('Output')
    plt.show()



    retval, mask = cv2.findHomography(pts1, pts2)
    print(retval, mask)
    M = cv2.estimateRigidTransform(pts1, pts2, True)
    print(M)

    dst = cv2.warpAffine(img1, M, (img1.shape[1], img1.shape[0]))
    #print(dst)

    plt.subplot(131), plt.imshow(img1), plt.title('img1')
    plt.subplot(132), plt.imshow(img2), plt.title('img2')
    plt.subplot(133), plt.imshow(dst), plt.title('img1 warped')
    plt.show()

    exit(0)

    '''

    pts1 = np.array([
        [437, 320],
        [91, 1790],
        [2220, 546],
        [2584, 1705],
        [1361, 1284],
        [2412, 1092],
        [2430, 1578]
    ]).reshape(-1, 1, 2)
    pts2 = np.array([
        [487, 494],
        [97, 1685],
        [2293, 420],
        [2603, 1766],
        [12294, 1277],
        [2398, 1028],
        [2400, 1601]
    ]).reshape(-1, 1, 2)

    #Normalize points:
    #pts1 = cv2.undistortPoints(pts1, intrinsics, distCoeffs=None, P=intrinsics)
    #pts2 = cv2.undistortPoints(pts2, intrinsics, distCoeffs=None, P=intrinsics)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    print('F (opencv):\n', F)
    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    #F = compute_fundamental(pts1, pts2)

    print('F:\n', F)
    print('F.T:\n', F.T)
    e1 = compute_epipole(F)
    e1 = np.dot(intrinsics, e1)
    e2 = compute_epipole(F.T)
    e2 = np.dot(intrinsics, e2)
    print('e1:\n', e1)
    print('e2:\n', e2)

    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    print('lines1: ', lines1)
    print('lines2: ', lines2)
    #img1[:,:] = np.fliplr(img1[:,:])
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    #img5[:,:] = np.fliplr(img5[:,:])
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    #plt.subplot(223), plt.imshow(img3)
    #plt.subplot(224), plt.imshow(img4)
    plt.show()

    E, mask = cv2.findEssentialMat(pts1, pts2, focal=focal, pp=(ppx, ppy))
    print('E: ', E)
    points, R, t, mask = cv2.recoverPose(E, pts1, pts2, focal=focal, pp=(ppx, ppy), mask=mask)
    print('R: ', R)
    print('t: ', t)
    M_r = np.hstack((R, t))
    M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

    print('angles:',rotationMatrixToEulerAngles(R))
    #return rotationMatrixToEulerAngles(R)


    #triangulate:

    P_l = np.dot(intrinsics, M_l)
    P_r = np.dot(intrinsics, M_r)
    point_4d_hom = cv2.triangulatePoints(M_l, M_r, pts1, pts2)
    print('point_4d_hom: ', point_4d_hom)
    point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
    point_3d = point_4d[:3, :].T

    print(point_3d)

    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D
    import random


    fig = pyplot.figure()
    ax = Axes3D(fig)

    sequence_containing_x_vals = point_3d[:,0]
    sequence_containing_y_vals = point_3d[:,1]
    sequence_containing_z_vals = point_3d[:,2]

    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals)
    pyplot.show()





def calibrate():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    rows = 7
    cols = 7
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    import pylab
    import imageio
    #filename = 'calib/checkerboard.mp4'
    #vid = imageio.get_reader(filename,  'ffmpeg')
    images = []
    #for i in range(10):
    #  image = vid.get_data(i)
    #  images.append(image)
    #  #cv2.imwrite('calib/frame_' + str(i) + '.jpg', image)
    import glob

    #for i in range(0, len(images), 1):
    #images = ['IMG_20171211_151001.jpg']
    images = glob.glob('calib/*.jpg')
    print(images)
    for fname in images:
        img = cv2.imread(fname)
        img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('img', gray)
        #cv2.waitKey(500)

        print('Processing calibration images')
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        print(ret)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (cols, rows), corners2, ret)
            #cv2.imshow('img', img)
            #cv2.waitKey(500)

    return  cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


def from_video():
    import pylab
    import imageio
    filename = 'toothpick.mp4'
    vid = imageio.get_reader(filename,  'ffmpeg')
    images = []
    for i in range(10): #350
      image = vid.get_data(i)
      images.append(image)

    return images
    #imgL = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)#cv2.imread('tsukuba_l.png',0)
    #imgR = cv2.cvtColor(images[150], cv2.COLOR_BGR2GRAY)#cv2.imread('tsukuba_r.png',0)

    #plt.figure()
    #plt.imshow(imgL, 'gray')
    #plt.figure()
    #plt.imshow(imgR, 'gray')
    #plt.show()


ret, mtx, dist, rvecs, tvecs = calibrate()
print('Intrinsic Matrix:\n', mtx)

#imgL = cv2.imread('im1.jpg')
#imgR = cv2.imread('im2.jpg')

#imgL = cv2.imread('img1/IMG_20171217_113125.jpg')
#imgR = cv2.imread('img1/IMG_20171217_113132.jpg')

imgL = cv2.imread('img2/100_7101.jpg')
imgR = cv2.imread('img2/100_7106.jpg')

imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2RGB)

match_features(imgL, imgR, mtx, dist)

exit(0)

angles = []
images = from_video()
imgL = images[0]
imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
for i, img in enumerate(images):
    if i % 10 != 1:
        continue
    imgR = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    angle = match_features(imgL, imgR, mtx, dist)
    print(angle)
    angles.append(angle)


angles = np.array(angles)
plt.plot(angles[:,0], 'r-', angles[:,1], 'g-', angles[:,2], 'b-')
plt.show()



img = cv2.imread('calib/IMG_20171211_151001.jpg')
img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(gray,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.png',dst)
plt.figure()
plt.imshow(dst, 'gray')
plt.show()

imgL = cv2.cvtColor(cv2.imread('im1.png'), cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(cv2.imread('im2.png'), cv2.COLOR_BGR2GRAY)

imgL = cv2.remap(imgL,mapx,mapy,cv2.INTER_LINEAR)
imgR = cv2.remap(imgR,mapx,mapy,cv2.INTER_LINEAR)


stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.figure()
plt.imshow(disparity,'gray')
print(disparity)
plt.show()