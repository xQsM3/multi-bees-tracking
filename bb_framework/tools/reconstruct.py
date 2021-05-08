# Luc Stiemer 2020
# THIS SCRIPT RECONSTRUCTS 3D FLIGHT PATH OUT OF A STEREO PAIR

# References:
#    [1]:    Multiple View Geometry, Richard Hartley and Andrew Zisserman
#    [2]:    OpenCV 4 with Python BluePrints, Menua Gevorgyan




#import python modules
import cv2 as cv
import numpy as np
import sys
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
#import own modules
from log import info

class SceneReconstruction3D:
    """3D scene reconstruction

        This class implements an algorithm for 3D scene reconstruction using
        stereo vision

        A 3D scene is reconstructed from a pair of images that show the same
        real-world scene from two different viewpoints. Feature matching is
        performed either with rich feature descriptors or based on optic flow.
        3D coordinates are obtained via triangulation.

        Note that a complete structure-from-motion pipeline typically includes
        bundle adjustment and geometry fitting, which are out of scope for
        this project.
    """

    def __init__(self,K1,dist1,K2,dist2,R,T,imageSize):
        """Constructor

            This method initializes the scene reconstruction algorithm.

            :param K: 3x3 intrinsic camera matrix
            :param dist: vector of distortion coefficients
        """
        self.K1 = K1
        self.K1_inv = np.linalg.inv(K1)  # store inverse for fast access
        self.d1 = dist1
        self.K2 = K2
        self.K2_inv = np.linalg.inv(K2)
        self.d2 = dist2
        self.R = R
        self.T = T
        self.imageSize = imageSize
        self.image_width = imageSize[1]
        self.image_height = imageSize[0]

    def load_image_pair(
            self,
            img_path1: str,
            img_path2: str,
            use_pyr_down: bool = True) -> None:
        
        img1_dist = self.load_image(img_path1,use_pyr_down)
        img2_dist = self.load_image(img_path2, use_pyr_down)


        if True:
            scaleFactor = max(self.imageSize) / max(img1_dist.shape)
            self.K1 = self.K1//scaleFactor
            self.K2 = self.K2//scaleFactor
            self.K1[2,2] = 1
            self.K2[2,2] = 1
            
        self.img1 = cv.undistort(img1_dist,self.K1,self.d1)
        self.img2 = cv.undistort(img2_dist, self.K2, self.d2)
        
        self.d1 = np.array([0,0,0,0,0]).reshape(1,5)
        self.d2 = np.array([0,0,0,0,0]).reshape(1,5)

    @staticmethod
    def load_image(
            img_path: str,
            use_pyr_down: bool,
            target_width: int = 600) -> np.ndarray:
        """Loads pair of images

            This method loads the two images for which the 3D scene should be
            reconstructed. The two images should show the same real-world scene
            from two different viewpoints.

            :param img_path1: path to first image
            :param img_path2: path to second image
            :param use_pyr_down: flag whether to downscale the images to
                                 roughly 600px width (True) or not (False)
        """

        img = cv.imread(img_path)
        
        # make sure image is valid
        assert img is not None, f"Image {img_path} could not be loaded."
        if len(img.shape) == 2:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        
        # scale down image if necessary
        while use_pyr_down and img.shape[1] > 2 * target_width:
            img = cv.pyrDown(img)
            
        return img

    def plot_optic_flow(self):
        """Plots optic flow field

            This method plots the optic flow between the first and second
            image.
        """
        self._extract_keypoints_flow()

        img = np.copy(self.img1)
        for pt1, pt2 in zip(self.match_pts1, self.match_pts2):
            cv.arrowedLine(img, tuple(pt1), tuple(pt2),
                            color=(255, 0, 0))

        cv.imshow("imgFlow", img)
        cv.waitKey()
        cv.destroyAllWindows()

    def draw_epipolar_lines(self, feat_mode: str = "flow"):
        """Draws epipolar lines

            This method computes and draws the epipolar lines of the two
            loaded images.

            :param feat_mode: whether to use rich descriptors for feature
                              matching ("sift") or optic flow ("flow")
        """
        self._extract_keypoints(feat_mode)
        self._find_fundamental_matrix()
        # Find epilines corresponding to points in right image (second image)
        # and drawing its lines on left image
        pts2 = self.match_pts2.reshape(-1, 1, 2)
        

        
        lines1 = get_epipolar_lines(pts2)
        
        img3, img4 = self._draw_epipolar_lines_helper(self.img1, self.img2,
                                                      lines1, self.match_pts1,
                                                      self.match_pts2)

        # Find epilines corresponding to points in left image (first image) and
        # drawing its lines on right image
        pts1re = self.match_pts1.reshape(-1, 1, 2)
        lines2 = cv.computeCorrespondEpilines(pts1re, 1, self.F)
        lines2 = lines2.reshape(-1, 3)
        img1, img2 = self._draw_epipolar_lines_helper(self.img2, self.img1,
                                                      lines2, self.match_pts2,
                                                      self.match_pts1)

        cv.imshow("left", img1)
        cv.imshow("right", img3)
        cv.waitKey()
        cv.destroyAllWindows()
    def get_epipolar_lines(self,pts2):
        pts2 = pts2.reshape(-1,1,2)
        #get fundamental matrix
        self._find_fundamental_matrix_via_projection_matrix()
        lines1 = cv.computeCorrespondEpilines(pts2, 2, self.F)
        lines1 = lines1.reshape(-1, 3)
        return lines1
    def plot_rectified_images(self, feat_mode: str = "flow",extrinsic_mode ="calib"):
        """Plots rectified images

            This method computes and plots a rectified version of the two
            images side by side.

            :param feat_mode: whether to use rich descriptors for feature
                              matching ("sift") or optic flow ("flow")
            :param extrinsic_mode: whether to use chessboard calibration "calib" 
                                    or feature matching for extrinsics                                        
        """
        #perform rectification
        R1, R2, P1, P2, Q, roi1, roi2 = self._perform_rectification()
        
        #check if vertical stereo
        vertical_stereo = self._check_for_vertical_stereo(P2)
            
        mapx1, mapy1 = cv.initUndistortRectifyMap(self.K1, self.d1, R1, P1,
                                                   self.img1.shape[:-1],
                                                   cv.CV_32F)
        mapx2, mapy2 = cv.initUndistortRectifyMap(self.K2, self.d2, R2, P2,
                                                   self.img1.shape[:-1],
                                                   cv.CV_32F)
        if vertical_stereo:
            img_rect1 = cv.transpose(cv.remap(self.img1, mapx1, mapy1, cv.INTER_LINEAR))
            img_rect2 = cv.transpose(cv.remap(self.img2, mapx2, mapy2, cv.INTER_LINEAR))
            img_rect1 = cv.flip(img_rect1,flipCode=0)
            img_rect2 = cv.flip(img_rect2,flipCode=0)
            
        else:
            img_rect1 = cv.remap(self.img1, mapx1, mapy1, cv.INTER_LINEAR)
            img_rect2 = cv.remap(self.img2, mapx2, mapy2, cv.INTER_LINEAR)
            
        cv.rectangle(img_rect1,roi1[:2],roi1[2:],(0,255,0),3)
        cv.rectangle(img_rect2,roi2[:2],roi1[2:],(0,255,0),3)
        # draw the images side by side
        total_size = (max(img_rect1.shape[0], img_rect2.shape[0]),
                      img_rect1.shape[1] + img_rect2.shape[1], 3)
        img = np.zeros(total_size, dtype=np.uint8)
        img[:img_rect1.shape[0], :img_rect1.shape[1]] = img_rect1
        img[:img_rect2.shape[0], img_rect1.shape[1]:] = img_rect2
        
        # draw horizontal lines every 25 px accross the side by side image
        for i in range(20, img.shape[0], 25):
            cv.line(img, (0, i), (img.shape[1], i), (255, 0, 0))
        cv.imshow('imgRectified', img)
        cv.waitKey()
        cv.destroyAllWindows()
        print("\nP1:\n{0}\nP2:\n{1}".format(P1,P2))
        
    def plot_point_cloud(self, feat_mode="flow"):
        """Plots 3D point cloud

            This method generates and plots a 3D point cloud of the recovered
            3D scene.

            :param feat_mode: whether to use rich descriptors for feature
                              matching ("sift") or optic flow ("flow")
        """
        self._extract_keypoints(feat_mode)
        self._find_fundamental_matrix()
        self._find_essential_matrix()
        self._find_camera_matrices_rt()

        # triangulate points
        first_inliers = np.array(self.match_inliers1)[:, :2]
        second_inliers = np.array(self.match_inliers2)[:, :2]
        pts4D = cv.triangulatePoints(self.Rt1, self.Rt2, first_inliers.T,
                                      second_inliers.T).T
        
        # convert from homogeneous coordinates to 3D
        pts3D = pts4D[:, :3] / pts4D[:, 3, None]

        # plot with matplotlib
        Xs, Zs, Ys = [pts3D[:, i] for i in range(3)]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(Xs, Ys, Zs, c=Ys,cmap=cm.hsv, marker='o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('3D point cloud: Use pan axes button below to inspect')
        plt.show()
        
    def select_2ImgPoints_And_Compute_Distance(self, feat_mode="flow",extrinsic_mode ="calib"):
        if extrinsic_mode == "matching":
            self._extract_keypoints(feat_mode)
            self._find_fundamental_matrix()
            self._find_essential_matrix()
            self._find_camera_matrices_rt()

            R = self.Rt2[:, :3]
            T = self.Rt2[:, 3]
            F = self.F
        elif extrinsic_mode == "calib":
            self._find_fundamental_matrix_via_projection_matrix()
            R = self.R
            T = self.T
            F = self.F
        
        P1,P2 = self._find_unrectified_projection_matrices(R,T)
        
        pts1,colorList = self._select_Img_Points_On_Frame(self.img1)
        
        epilines2 = cv.computeCorrespondEpilines(pts1,1,F)
        epilines2 = epilines2.reshape(-1,3)

        img2,colorList = self._draw_lines(self.img2,epilines2,colorList)
        
        pts2 = self._select_Img_Points_On_Epiline(img2,epilines2,colorList)

        pointDistance = self._compute_euclidean_distance_between_points(pts1,pts2,P1,P2)
        
        print('[INFO:] point distance: {0}'.format(pointDistance))
        
    def triangulate_tracks(self,sequence1,sequence2,sequence3D):
        # prepare point arrays for triangulation, fill empty frame_idx with np.nans
        for id1,row in enumerate(sequence3D.match_matrix):
            for id2,el in enumerate(row):
                if el == True:
                    track1 = sequence1.tracks[sequence1.tracks[:,1]==id1]
                    for frame_idx in range(sequence1.min_frame_idx,sequence1.max_frame_idx+1):
                        if len(track1[track1[:,0]==frame_idx]) == 0:
                            if 'pts1' in locals():
                                pts1 = np.vstack((pts1,np.array([[np.nan,np.nan]])))
                            else:
                                pts1 = np.array([[np.nan,np.nan]])

                        else:
                            if 'pts1' in locals():
                                pts1 = np.vstack((pts1,track1[track1[:,0]==frame_idx][0,2:4]))
                            else:
                                pts1 = np.array([track1[track1[:,0]==frame_idx][0,2:4]])
                    
                    track2 = sequence2.tracks[sequence2.tracks[:,1]==id2]
                    
                    for frame_idx in range(sequence2.min_frame_idx,sequence2.max_frame_idx+1):
                        if len(track2[track2[:,0]==frame_idx]) == 0:
                            if 'pts2' in locals():
                                pts2 = np.vstack((pts2,np.array([[np.nan,np.nan]])))
                            else:
                                pts2 = np.array([[np.nan,np.nan]])
                        else:
                            if 'pts2' in locals():
                                pts2 = np.vstack((pts2,track2[track2[:,0]==frame_idx][0,2:4]))
                            else:
                                pts2 = np.array([track2[track2[:,0]==frame_idx][0,2:4]])

                    pts3D = self._triangulate(pts1,pts2,extrinsic_mode ="calib")
                    # clear memory
                    del pts1
                    del pts2
                    
                    #construct 3d tracks out of pts3D
                    for frame_idx,pt in enumerate(pts3D):
                        if np.isnan(pt).any():
                            continue
                        track_pt = np.array([[frame_idx,id1,pt[0,0],pt[0,1],pt[0,2]]])
                        if 'tracks' in locals():
                            tracks = np.vstack((tracks,track_pt))
                        else:
                            tracks = track_pt
        if 'tracks' in locals():
            # sort tracks array for frame_idx column            
            tracks = tracks[tracks[:,0].argsort()]
            return tracks
        else:
            return np.nan
    def _check_for_vertical_stereo(self,P2):
        #checks if stereo pair is vertical or horizontal
        #look up OpenCV docu for this
        if abs(P2[1][3])>abs(P2[0][3]):
            return True
        else:
            return False   
        
    def _compute_euclidean_distance_between_points(self,pts1,pts2,P1,P2):
        #computes point distance for calib validation
        
        #estimate 3D coordinates out of stereo pair's 2D coordinates
        points3Dh = cv.triangulatePoints(P1,P2,
                                         pts1.T,pts2.T,
                                         None)
        #convert coordinates from homogenous in cartesian
        point1_3D = cv.convertPointsFromHomogeneous(points3Dh[:,0].reshape(1,4))
        point2_3D = cv.convertPointsFromHomogeneous(points3Dh[:,1].reshape(1,4))
        #compute distance
        pointDistance = math.sqrt( (point1_3D[0][0][0]-point2_3D[0][0][0])**2 
                                  +(point1_3D[0][0][1]-point2_3D[0][0][1])**2 
                                  +(point1_3D[0][0][2]-point2_3D[0][0][2])**2 )
        return pointDistance

    def _draw_epipolar_lines_helper(self, img1, img2, lines, pts1, pts2):
        """
        Helper method to draw epipolar lines and features
        """
        if img1.shape[2] == 1:
            img1 = cv.cvtColor(img1, cv.COLOR_GRAY2BGR)
        if img2.shape[2] == 1:
            img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

        c = img1.shape[1]
        for r, pt1, pt2 in zip(lines, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2] / r[1]])
            x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
            cv.line(img1, (x0, y0), (x1, y1), color, 1)
            cv.circle(img1, tuple(pt1), 5, color, -1)
            cv.circle(img2, tuple(pt2), 5, color, -1)
        return img1, img2
    
    def _draw_lines(self,img2,lines2,colorList):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        _,c,_ = img2.shape
        
        for i,r in enumerate(lines2):
            color = colorList[i]
            x0,y0 = map(int, [0, -r[2]/r[1] ])
            x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
            img2 = cv.line(img2, (x0,y0), (x1,y1), color,1)
            colorList.append(color)
        return img2,colorList        
        
    def _extract_keypoints(self, feat_mode):
        """Extracts keypoints

            This method extracts keypoints for feature matching based on
            a specified mode:
            - "sift": use rich sift descriptor
            - "flow": use optic flow

            :param feat_mode: keypoint extraction mode ("sift" or "flow")
        """
        # extract features
        if feat_mode.lower() == "sift":
            # feature matching via sift and BFMatcher
            self._extract_keypoints_sift()
        elif feat_mode.lower() == "flow":
            # feature matching via optic flow
            self._extract_keypoints_flow()
        else:
            sys.exit(f"Unknown feat_mode {feat_mode}. Use 'sift' or 'FLOW'")

    def _extract_keypoints_sift(self):
        """Extracts keypoints via sift descriptors"""
        # extract keypoints and descriptors from both images
        # detector = cv.xfeatures2d.SIFT_create(contrastThreshold=0.11, edgeThreshold=10)
        detector = cv.xfeatures2d.SIFT_create()
        first_key_points, first_desc = detector.detectAndCompute(self.img1,
                                                                 None)
        second_key_points, second_desc = detector.detectAndCompute(self.img2,
                                                                   None)
        # match descriptors
        matcher = cv.BFMatcher(cv.NORM_L1, True)
        matches = matcher.match(first_desc, second_desc)

        # generate lists of point correspondences
        self.match_pts1 = np.array(
            [first_key_points[match.queryIdx].pt for match in matches])
        self.match_pts2 = np.array(
            [second_key_points[match.trainIdx].pt for match in matches])

    def _extract_keypoints_flow(self):
        """Extracts keypoints via optic flow"""
        # find FAST features
        fast = cv.FastFeatureDetector_create()
        first_key_points = fast.detect(self.img1)

        first_key_list = [i.pt for i in first_key_points]
        first_key_arr = np.array(first_key_list).astype(np.float32)

        second_key_arr, status, err = cv.calcOpticalFlowPyrLK(
            self.img1, self.img2, first_key_arr, None)

        # filter out the points with high error
        # keep only entries with status=1 and small error
        condition = (status == 1) * (err < 5.)
        concat = np.concatenate((condition, condition), axis=1)
        first_match_points = first_key_arr[concat].reshape(-1, 2)
        second_match_points = second_key_arr[concat].reshape(-1, 2)

        self.match_pts1 = first_match_points
        self.match_pts2 = second_match_points

    def _find_fundamental_matrix(self):
        """Estimates fundamental matrix """
        self.F, self.Fmask = cv.findFundamentalMat(self.match_pts1,
                                                    self.match_pts2,
                                                    cv.FM_RANSAC, 0.1, 0.99)
    def _find_fundamental_matrix_via_projection_matrix(self):
        #this function is based on method [1, page244]
        C = np.array([0,0,0,1]).reshape(4,1)
        I = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
        I0 = np.append(I,np.array([0,0,0]).reshape(3,1),axis=1)
        Rt = np.append(self.R,self.T,axis=1)
        P1ur = self.K1.dot(I0)
        P2ur = self.K2.dot(Rt)
        e2 = P2ur.dot(C)
        e2skew = np.array([0,0,0,0,0,0,0,0,0]).reshape(3,3)
        e2skew[0,1],e2skew[0,2],e2skew[1,0],e2skew[1,2],e2skew[2,0],e2skew[2,1] = -e2[2],e2[1],e2[2],-e2[0],-e2[1],e2[0]
        P1urPlus = np.append(np.linalg.inv(P1ur[:,0:3]),np.array([0,0,0]).reshape(1,3),axis=0)
        F = e2skew.dot(P2ur).dot(P1urPlus)
        F = F/F[2,2]
        self.F = F
        
    def _find_essential_matrix(self):
        """Estimates essential matrix based on fundamental matrix """
        self.E = self.K2.T.dot(self.F).dot(self.K1)

    def _find_camera_matrices_rt(self):
        """Finds the [R|t] camera matrix"""
        # decompose essential matrix into R, t (See [1] 9.13)
        U, S, Vt = np.linalg.svd(self.E)
        W = np.array([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                      1.0]).reshape(3, 3)

        # iterate over all point correspondences used in the estimation of the
        # fundamental matrix
        first_inliers = []
        second_inliers = []
        for pt1, pt2, mask in zip(
                self.match_pts1, self.match_pts2, self.Fmask):
            if mask:
                # normalize and homogenize the image coordinates
                first_inliers.append(self.K1_inv.dot([pt1[0], pt1[1], 1.0]))
                second_inliers.append(self.K2_inv.dot([pt2[0], pt2[1], 1.0]))

        # Determine the correct choice of second camera matrix
        # only in one of the four configurations will all the points be in
        # front of both cameras

        R = T = None
        R = U.dot(W.T).dot(Vt)
        T = U[:, 2]
        for r in (U.dot(W).dot(Vt), U.dot(W.T).dot(Vt)):
            for t in (U[:, 2], -U[:, 2]):
                if self._in_front_of_both_cameras(
                        first_inliers, second_inliers, r, t):
                    R, T = r, t

        assert R is not None, "Camera matricies were never found"

        self.match_inliers1 = first_inliers
        self.match_inliers2 = second_inliers
        self.Rt1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        self.Rt2 = np.hstack((R, T.reshape(3, 1)))
        
    def _find_unrectified_projection_matrices(self,R,T):
        #compute unrectified projection matrices
        R1 = np.array([1,0,0,0,1,0,0,0,1]).reshape(3,3)
        T1 = np.array([0,0,0]).reshape(3,1)
        P1 = np.dot(self.K1,np.append(R1,T1,axis=1))
        P2= np.dot(self.K2,np.append(R,T,axis=1))
        return P1,P2

    def _in_front_of_both_cameras(self, first_points, second_points, rot,
                                  trans):
        """Determines whether point correspondences are in front of both
           images"""
        print("start")
        rot_inv = rot
        for first, second in zip(first_points, second_points):
            first_z = np.dot(rot[0, :] - second[0] * rot[2, :],
                             trans) / np.dot(rot[0, :] - second[0] * rot[2, :],
                                             second)
            first_3d_point = np.array([first[0] * first_z,
                                       second[0] * first_z, first_z])
            second_3d_point = np.dot(rot.T, first_3d_point) - np.dot(rot.T,
                                                                     trans)

            #print(first_3d_point,second_3d_point)
            if first_3d_point[2] < 0 or second_3d_point[2] < 0:
                return False

        return True

    def _linear_ls_triangulation(self, u1, P1, u2, P2):
        """Triangulation via Linear-LS method"""
        # build A matrix for homogeneous equation system Ax=0
        # assume X = (x,y,z,1) for Linear-LS method
        # which turns it into AX=B system, where A is 4x3, X is 3x1 & B is 4x1
        A = np.array([u1[0] * P1[2, 0] - P1[0, 0], u1[0] * P1[2, 1] - P1[0, 1],
                      u1[0] * P1[2, 2] - P1[0, 2], u1[1] * P1[2, 0] - P1[1, 0],
                      u1[1] * P1[2, 1] - P1[1, 1], u1[1] * P1[2, 2] - P1[1, 2],
                      u2[0] * P2[2, 0] - P2[0, 0], u2[0] * P2[2, 1] - P2[0, 1],
                      u2[0] * P2[2, 2] - P2[0, 2], u2[1] * P2[2, 0] - P2[1, 0],
                      u2[1] * P2[2, 1] - P2[1, 1],
                      u2[1] * P2[2, 2] - P2[1, 2]]).reshape(4, 3)

        B = np.array([-(u1[0] * P1[2, 3] - P1[0, 3]),
                      -(u1[1] * P1[2, 3] - P1[1, 3]),
                      -(u2[0] * P2[2, 3] - P2[0, 3]),
                      -(u2[1] * P2[2, 3] - P2[1, 3])]).reshape(4, 1)

        ret, X = cv.solve(A, B, flags=cv.DECOMP_SVD)
        return X.reshape(1, 3)

    def _perform_rectification(self,extrinsic_mode ="calib"):
        '''
        computes rectification
        cv.stereoRectify appears to be bugged, triangulation will not work with
        rectified projection matrices, but these matrices are still usefull later for
        path evaluation. Since after rectification epilines are horizontal, its easy
        to compare image points between stereo pairs 
        '''
        if extrinsic_mode == "matching":
            self._extract_keypoints(feat_mode)
            self._find_fundamental_matrix()
            self._find_essential_matrix()
            self._find_camera_matrices_rt()

            R = self.Rt2[:, :3]
            T = self.Rt2[:, 3]
        elif extrinsic_mode == "calib":
            self._find_fundamental_matrix_via_projection_matrix()
            R = self.R
            T = self.T
            F = self.F
        
        # perform the rectification
        R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(self.K1, self.d1,
                                                          self.K2, self.d2,
                                                          self.imageSize,
                                                          R, T, flags = 0,alpha=-1.0)
        return R1, R2, P1, P2, Q, roi1, roi2
    
    def _rectify_tracks(self,track1,track2,R1,R2,P1,P2):
        '''
        bugged in current opencv version 4.4 dude to bug in stereoRectify
        '''
        #remap image points to rectified plane
        pts1 = np.array([[track1[:,2],track1[:,3]]])
        pts2 = np.array([[track2[:,2],track2[:,3]]])
        
        rectrack1 = track1.copy()
        rectrack2 = track2.copy()

        pts1=pts1.astype(np.float32).reshape(-1, 2)
        pts2=pts2.astype(np.float32).reshape(-1, 2)
        
        recpts1 = cv.undistortPoints(pts1,self.K1,self.d1,None,R1,P1)
        recpts2 = cv.undistortPoints(pts2,self.K2,self.d2,None,R2,P2)
        rectrack1[:,2],rectrack1[:,3] = recpts1.T[0],recpts1.T[1]
        rectrack2[:,2],rectrack2[:,3] = recpts2.T[0],recpts2.T[1]
        return rectrack1,rectrack2
    
    def _select_Img_Points_On_Frame(self,frame):
        #helper function to select points on frame for distance computation
        points = [[]]
        colorList = []
        #defines a function which will draw a point
        def drawPoint(event,mx,my,flags,param):
            nonlocal points,colorList
            #start drawing if left mouse button is clicked
            if event == cv.EVENT_LBUTTONDOWN:
                color = tuple(np.random.randint(0,255,3).tolist())
                colorList.append(color)
                if points == [[]]:
                    cv.circle(frame,(mx,my),6,color,-1)
                    points = np.asarray([mx,my],dtype=np.float64).reshape(1,2)
                else:
                    cv.circle(frame,(mx,my),6,color,-1)
                    mx,my = np.float64(mx),np.float64(my)
                    points = np.append(points,[[mx,my]],axis=0)
                    

        cv.namedWindow('mask')
        #connects the window, the drawRec function, and the mouse
        cv.setMouseCallback('mask',drawPoint)
        
        info.info('select the two points you want to compute the distance of, then press enter')
        #shows the image until user hits escape
        while True:
            cv.imshow('mask',frame)

            keyboard = cv.waitKey(20)
            if keyboard & 0xFF == 13:
                break

        cv.destroyAllWindows()
        #print('points1:\n{0}\npoints1.T:\n{1}'.format(points,points.T))
        return points,colorList

    def _select_Img_Points_On_Epiline(self,frame,epilines,colorList):
        '''
        this function is for calib validation
        displays image of camera 2, displays epilines of the points selected in camera 1 and
        asks user to mark the point in camera 2
        '''
        points = [[]]
        i = 0
        #defines a function which will draw the rectangle on the image
        def drawPoint(event,mx,my,flags,param):
            nonlocal points,i
            #start drawing if left mouse button is clicked
            if event == cv.EVENT_LBUTTONDOWN:
                cv.line(frame,(mx,0),(mx,frame.shape[0]),colorList[i])
                epi_y = int((-epilines[i][0]*mx-epilines[i][2])/epilines[i][1])
                
                i +=1
                if i==len(colorList):
                    i -=1
                if points == [[]]:
                    print('create')
                    points = np.array([mx,epi_y],dtype=np.float64).reshape(1,2)  
                else:
                    mx,my = np.float64(mx),np.float64(my)
                    points = np.append(points,[[mx,epi_y]],axis=0)
                    print('append')

        #define the window size
        
        cv.namedWindow('mask')
        #connects the window, the drawRec function, and the mouse
        cv.setMouseCallback('mask',drawPoint)
        
        info.info('''if your selected points does not lay perfectly,
            on the displayed epilines in the second camera,
            something is probably wrong with the calibration''')
        info.info('select the same two image points, but this time in the second camera\nthen press escape')
        #shows the image until user hits escape
        while True:
            cv.imshow('mask',frame)

            if cv.waitKey(20) & 0xFF == 27:
                break

        cv.destroyAllWindows()
        #print('points:\n{0}\npoints.T:\n{1}'.format(points,points.T))
        return points
    
    def _triangulate(self,pts1,pts2,extrinsic_mode ="calib"):
        """
    .   helper function for triangulate_tracks
    .   uses unrectified projection matrices, since cv.stereoRectify appears to be bugged
    """
        if extrinsic_mode == "matching":
            self._extract_keypoints(feat_mode)
            self._find_fundamental_matrix()
            self._find_essential_matrix()
            self._find_camera_matrices_rt()

            R = self.Rt2[:, :3]
            T = self.Rt2[:, 3]
            F = self.F
        elif extrinsic_mode == "calib":
            R = self.R
            T = self.T
            
        

        #get projection matrices, since rectification in OpenCV appears
        #to be bugged, unrectified projection is used here
        P1,P2 = self._find_unrectified_projection_matrices(R,T)
        
        
        
        #share nan values between pts1 and pts2
        pts1_sharedNaN = pts1.copy().astype(float)
        pts2_sharedNaN = pts2.copy().astype(float)
        pts1_sharedNaN[np.isnan(pts2)] = np.nan
        pts2_sharedNaN[np.isnan(pts1)] = np.nan
        
        
        #remove nans from point arrays since cv.triangulatePoints cannot handle nan values
        pts1_noNaN = pts1_sharedNaN[~np.isnan(pts1_sharedNaN).any(axis=1)]
        pts2_noNaN = pts2_sharedNaN[~np.isnan(pts2_sharedNaN).any(axis=1)]
        
        #perform triangulation
        points3Dh = cv.triangulatePoints(P1,P2,
                                        pts1_noNaN.T,pts2_noNaN.T,
                                        None)     
        points3D = cv.convertPointsFromHomogeneous(points3Dh.T)
        
        #refill 3D points with nan values
        for i,point in enumerate(pts1_sharedNaN):
            if np.isnan(point).any():
                points3D = np.insert(points3D,i,[np.nan,np.nan,np.nan],axis=0)
                
        return points3D