import cv2
import pickle
import os
import numpy as np


class Aruco:
    def __init__(self, dict_name: str = "DICT_ARUCO_ORIGINAL") -> None:
        self.dict_name = dict_name
        with open(os.path.join("./dict", self.dict_name + ".pickle"), "rb") as f:
            self.marker_size, self.n_markers, self.dict = pickle.load(f)

    def detect(self, img_gray, block_size=21, C=2):
        # Will return an array of (id, corners)

        # Create edge image using adaptive threshold
        img_thresh = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, block_size, C)
        contours, _ = cv2.findContours(
            img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_rects = []

        for cnt in contours:
            cnt = cv2.approxPolyDP(cnt, epsilon=5, closed=True)
            # Detect rectangles of decent size
            if len(cnt) == 4 and cv2.contourArea(cnt) > 200:
                candidate_rects.append(cnt)

        aruco_coords = np.array([[[40, 0]], [[0, 0]], [[0, 40]], [[40, 40]]])

        result = []

        for rect in candidate_rects:
            h, status = cv2.findHomography(rect, aruco_coords)
            img_out = cv2.warpPerspective(img_gray, h, dsize=(40, 40))
            img_out = cv2.resize(img_out, (self.marker_size, self.marker_size))
            ret, img_out = cv2.threshold(img_out, 127, 255, cv2.THRESH_BINARY)
            img_out //= 255

            # If it is valid then add it to results
            if tuple(img_out.ravel().tolist()) in self.dict:
                res, rot = self.dict[tuple(img_out.ravel())]
                if rot == 1:
                    rect = np.concatenate([rect[1:], rect[0:1]], axis=0)
                if rot == 2:
                    rect = np.concatenate([rect[2:], rect[0:2]], axis=0)
                if rot == 3:
                    rect = np.concatenate([rect[3:], rect[0:3]], axis=0)
                result.append(
                    [res,
                     rect]
                )
        return result

    def estimate_pose_from_single_marker(self, corners, size, camera_matrix, dist_coeffs):
        obj_pts = np.array(
            [
                [size/2, size/2, 0],
                [-size/2, size/2, 0],
                [-size/2, -size/2, 0],
                [size/2, -size/2, 0]
            ], dtype=np.float32
        )

        ret, rvec, tvec = cv2.solvePnP(obj_pts, corners.reshape(
            (4, 2)).astype(float), camera_matrix, dist_coeffs)
        return ret, rvec, tvec

    def draw_axis(self, img, camera_matrix, dist_coeff, rvec, tvec, axis_size=0.4):

        obj_pts = np.array(
            [
                [0, 0, 0],
                [axis_size, 0, 0],
                [0, axis_size, 0],
                [0, 0, axis_size]
            ], dtype=np.float32
        )
        points, _ = cv2.projectPoints(
            obj_pts, rvec, tvec, cameraMatrix=camera_matrix, distCoeffs=dist_coeff)
        points = points.astype(int).reshape((4, 2))
        cv2.line(img, points[0], points[1], (0, 0, 255), 2)
        cv2.line(img, points[0], points[2], (0, 255, 0), 2)
        cv2.line(img, points[0], points[3], (255, 0, 0), 2)


if __name__ == "__main__":
    import time
    img = cv2.imread("./test_image.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco = Aruco("DICT_6X6_250")
    start = time.time()
    out = aruco.detect(img_gray)
    end = time.time()
    print(end-start)

    camera_matrix = np.array([
        [1430, 0, 480],
        [0, 1430, 620],
        [0, 0, 1]
    ], dtype=float)

    img_out = img.copy()

    for id, corners in out:
        x, y = np.mean(corners.squeeze(), axis=0).astype(int)
        cv2.polylines(img_out, [corners], True, (255, 0, 0), thickness=2)
        res, rvec, tvec = aruco.estimate_pose_from_single_marker(
            corners, 1, camera_matrix, None)
        aruco.draw_axis(img_out, camera_matrix, None, rvec, tvec)
    cv2.imshow("Input", img)
    cv2.imshow("Output", img_out)
    cv2.imwrite("./output_image.jpg", img_out)
    cv2.waitKey()
