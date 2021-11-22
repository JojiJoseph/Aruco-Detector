import cv2
import pickle
import os
import numpy as np


class Aruco:
    def __init__(self, dict_name="DICT_ARUCO_ORIGINAL") -> None:
        self.dict_name = dict_name
        with open(os.path.join("./dict", self.dict_name + ".pickle"), "rb") as f:
            self.marker_size, self.n_markers, self.dict = pickle.load(f)

    def detect(self, img_gray):
        # Will return an array of (id, corners)

        # Create edge image using adaptive threshold
        img_thresh = cv2.adaptiveThreshold(
            img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 0)
        contours, _ = cv2.findContours(
            img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_rects = []
        
        for cnt in contours:
            cnt = cv2.approxPolyDP(cnt, epsilon=5, closed=True)
            # Detect rectange of decent size
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
                result.append(
                    [self.dict[tuple(img_out.ravel())][0],
                     rect]
                )
        return result


if __name__ == "__main__":
    import time
    img = cv2.imread("./test_image.jpg")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco = Aruco("DICT_6X6_250")
    start = time.time()
    out = aruco.detect(img_gray)
    end =  time.time()
    print(end-start)

    img_out = img.copy()
    for id, corners in out:
        x, y = np.mean(corners.squeeze(), axis=0).astype(int)
        cv2.polylines(img_out, [corners], True, (255,0,0), thickness=2)
        img_out = cv2.drawMarker(img_out, (x, y), (0, 0, 255), thickness=2)
    cv2.imshow("Input", img)
    cv2.imshow("Output", img_out)
    cv2.imwrite("./output_image.jpg", img_out)
    cv2.waitKey()
