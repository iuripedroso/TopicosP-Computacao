import cv2
import numpy as np
from skimage.feature import hog

HOG_SIZE = (64, 128)

def extract_hog(roi):
    roi_r = cv2.resize(roi, HOG_SIZE)
    fd = hog(roi_r, orientations=9, pixels_per_cell=(8,8),
             cells_per_block=(2,2), feature_vector=True)
    return fd.astype(np.float32)

def extract_hu_moments(roi):
    _, bw = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
    m  = cv2.moments(bw)
    hu = cv2.HuMoments(m).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu.astype(np.float32)

def extract_lbp_hist(roi, radius=2, n_points=16):
    roi_r = cv2.resize(roi, (64, 64))
    h, w  = roi_r.shape
    lbp   = np.zeros((h, w), dtype=np.uint8)
    for i in range(radius, h - radius):
        for j in range(radius, w - radius):
            center = roi_r[i, j]
            code = 0
            neighbors = [
                roi_r[i-radius, j], roi_r[i-radius, j+radius],
                roi_r[i, j+radius], roi_r[i+radius, j+radius],
                roi_r[i+radius, j], roi_r[i+radius, j-radius],
                roi_r[i, j-radius], roi_r[i-radius, j-radius],
            ]
            for k, nb in enumerate(neighbors):
                if nb >= center:
                    code |= (1 << k)
            lbp[i, j] = code
    hist, _ = np.histogram(lbp.ravel(), bins=32, range=(0, 256))
    return hist.astype(np.float32)

def extract_descriptor(roi):
    hog_feat = extract_hog(roi)
    hu_feat  = extract_hu_moments(roi)
    lbp_feat = extract_lbp_hist(roi)
    return np.concatenate([hog_feat, hu_feat, lbp_feat])