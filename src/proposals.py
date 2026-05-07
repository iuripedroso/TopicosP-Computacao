import numpy as np

def sliding_window_proposals(img, scales=None, step=16, min_dark_ratio=0.01):
    if scales is None:
        scales = [(64, 128), (96, 192), (128, 256)]
    h, w = img.shape[:2]
    proposals = []
    for (ww, wh) in scales:
        if ww > w or wh > h:
            continue
        for y in range(0, h - wh + 1, step):
            for x in range(0, w - ww + 1, step):
                roi = img[y:y+wh, x:x+ww]
                dark_ratio = np.sum(roi < 128) / roi.size
                if dark_ratio > min_dark_ratio:
                    proposals.append((x, y, ww, wh))
    return proposals

def compute_iou(boxA, boxB):
    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = ax1 + boxA[2], ay1 + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = bx1 + boxB[2], by1 + boxB[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (union + 1e-6)

def nms_proposals(proposals, iou_thresh=0.5):
    if not proposals:
        return []
    boxes  = np.array(proposals, dtype=float)
    areas  = boxes[:, 2] * boxes[:, 3]
    order  = areas.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        x1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        y1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        x2 = np.minimum(boxes[i,0]+boxes[i,2], boxes[order[1:],0]+boxes[order[1:],2])
        y2 = np.minimum(boxes[i,1]+boxes[i,3], boxes[order[1:],1]+boxes[order[1:],3])
        inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thresh)[0] + 1]
    return [proposals[k] for k in keep]

def get_best_proposal(img, proposals):
    if not proposals:
        h, w = img.shape[:2]
        return (0, 0, w, h)
    areas = [p[2]*p[3] for p in proposals]
    return proposals[int(np.argmax(areas))]