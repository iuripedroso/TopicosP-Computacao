import os
import random
import numpy as np
import cv2

EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

def load_images_from_folder(folder, label, max_images=None):
    entries = []
    files = sorted([f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS)])
    if max_images:
        files = files[:max_images]
    for fname in files:
        path = os.path.join(folder, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        entries.append({
            'filename': fname,
            'path':     path,
            'label':    label,
            'img':      img,
        })
    return entries

def build_dataset(dataset_root, n_docs=25, n_queries=5):
    org_folder  = os.path.join(dataset_root, 'full_org')
    forg_folder = os.path.join(dataset_root, 'full_forg')

    if not os.path.exists(org_folder):
        raise FileNotFoundError(f"Pasta não encontrada: {org_folder}")
    if not os.path.exists(forg_folder):
        raise FileNotFoundError(f"Pasta não encontrada: {forg_folder}")

    org_imgs  = load_images_from_folder(org_folder,  'original')
    forg_imgs = load_images_from_folder(forg_folder, 'falsificada')

    all_imgs = org_imgs + forg_imgs
    random.shuffle(all_imgs)

    needed = n_docs + n_queries
    if len(all_imgs) < needed:
        raise ValueError(f"Dataset tem apenas {len(all_imgs)} imagens, precisa de pelo menos {needed}")

    selected = all_imgs[:needed]
    for i, entry in enumerate(selected):
        entry['id'] = i

    docs    = selected[:n_docs]
    queries = selected[n_docs:n_docs + n_queries]

    return docs, queries

def preprocess(img, target_size=(128, 256)):
    h, w = img.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.ones((th, tw), dtype=np.uint8) * 255
    y_off = (th - new_h) // 2
    x_off = (tw - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    binarized = cv2.adaptiveThreshold(
        canvas, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 15, 8
    )
    return binarized