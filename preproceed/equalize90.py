import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = r"train-ztd/2ST-14_1_A2-1-14_20260810002217.jpg"
OUT_DIR = "./test"
ROI = (0, 630, 1280, 1024)  # x, y, w, h
KEEP = 0.90               # 保留中间 90%,两端各 5% 舍去


def build_lut(crop, keep=KEEP):
    total = crop.size
    lo = int(total * (1 - keep) / 2)
    hi = int(total * (1 + keep) / 2)
    hist = cv2.calcHist([crop], [0], None, [256], [0, 256]).ravel().astype(np.int64)
    cdf = np.cumsum(hist)
    vmin = int(np.searchsorted(cdf, lo, side="left"))
    vmax = int(np.searchsorted(cdf, hi, side="right"))
    vmin = max(vmin, 0)
    vmax = min(vmax, 255)
    if vmax <= vmin:
        vmax = vmin + 1
    lut = np.zeros(256, dtype=np.uint8)
    x = np.arange(256)
    mapped = np.clip((x - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    lut[:vmin] = 0
    lut[vmin:vmax] = mapped[vmin:vmax]
    lut[vmax:] = 255
    return lut, vmin, vmax


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    img = cv2.imread(SRC, cv2.IMREAD_UNCHANGED)
    x, y, w, h = ROI
    crop = img[y:y + h, x:x + w].copy()

    lut, vmin, vmax = build_lut(crop)
    eq_crop = cv2.LUT(crop, lut)

    result = img.copy()
    result[y:y + h, x:x + w] = eq_crop

    cv2.imwrite(os.path.join(OUT_DIR, "roi_equalized.png"), eq_crop)
    cv2.imwrite(os.path.join(OUT_DIR, "full_equalized.png"), result)
    cv2.imwrite(os.path.join(OUT_DIR, "original.png"), img)
    cv2.imwrite(os.path.join(OUT_DIR, "roi.png"), crop)

    mark = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(mark, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(OUT_DIR, "region_marked.png"), mark)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    for ax, data, title in [
        (axs[0, 0], crop, "ROI original"),
        (axs[0, 1], eq_crop, "ROI equalized (90% clip)"),
        (axs[1, 0], result, "Full result"),
        (axs[1, 1], None, None),
    ]:
        if data is not None:
            ax.imshow(data, cmap="gray")
            ax.set_title(title)
            ax.axis("off")

    ax = axs[1, 1]
    ax.hist(crop.ravel(), bins=256, range=(0, 255), alpha=0.6, label="orig", color="b")
    ax.hist(eq_crop.ravel(), bins=256, range=(0, 255), alpha=0.6, label="eq", color="r")
    ax.axvline(vmin, color="g", ls="--", label=f"clip lo={vmin}")
    ax.axvline(vmax, color="g", ls="--", label=f"clip hi={vmax}")
    ax.set_title("Histogram (ROI)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "comparison.png"), dpi=120)
    plt.close()

    print(f"vmin={vmin}, vmax={vmax} (kept middle {KEEP:.0%})")
    print(f"saved -> {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
