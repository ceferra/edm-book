import numpy as np

def iou(a, b):
    """Intersection over union of two boxes given as (x1, y1, x2, y2)."""
    w = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return w * h / (area(a) + area(b) - w * h)

print(f"IoU = {iou((2, 2, 6, 6), (4, 3, 8, 7)):.3f}")

tp = np.array([1, 1, 0, 1, 0, 0, 1, 0])     # detections sorted by confidence
n_gt = 5                                    # number of ground-truth objects
prec = np.cumsum(tp) / np.arange(1, len(tp) + 1)
rec = np.cumsum(tp) / n_gt
env = np.maximum.accumulate(prec[::-1])[::-1]   # interpolated precision
ap = np.sum(np.diff(np.r_[0, rec]) * env)       # all-point interpolation
print("precision:", prec.round(3))
print("recall:   ", rec.round(2))
print(f"AP = {ap:.4f}")
