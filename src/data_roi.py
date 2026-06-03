import numpy as np


def crop_brain_roi(image, mask, threshold=5, padding=8):
    image_np = np.array(image)
    ys, xs = np.where(image_np > threshold)

    if len(xs) == 0 or len(ys) == 0:
        return image, mask

    h, w = image_np.shape
    left = max(int(xs.min()) - padding, 0)
    top = max(int(ys.min()) - padding, 0)
    right = min(int(xs.max()) + padding + 1, w)
    bottom = min(int(ys.max()) + padding + 1, h)

    box = (left, top, right, bottom)
    return image.crop(box), mask.crop(box)
