import numpy as np
from pathlib import Path

P = Path(__file__).resolve().parent
points = np.load(P / "centerline_points.npy")
if len(points) < 4:
    raise ValueError("Need closed centerline")
dense = [points[0]]
cum = [0.0]
total = 0.0

for a, b in zip(points[:-1], points[1:]):
    length = float(np.linalg.norm(b - a))
    for j in range(1, max(1, int(np.ceil(length))) + 1):
        n = max(1, int(np.ceil(length)))
        pt = a + (b - a) * (j / n)
        total += float(np.linalg.norm(pt - dense[-1]))
        dense.append(pt)
        cum.append(total)

if total <= 0:
    raise ValueError("Lap length is zero")

np.save(P / "dense_centerline.npy", np.array(dense[:-1]))
np.save(P / "cumulative_distance.npy", np.array(cum[:-1]))
np.save(P / "lap_length.npy", np.array(total))
print("Points:", len(dense) - 1, "Lap length:", round(total, 2))
