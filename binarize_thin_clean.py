
import cv2
import numpy as np
import argparse
from pathlib import Path
from skimage.morphology import thin


def binarize_strict(img_bgr, white_th=245):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bin01 = (gray < white_th).astype(np.uint8)
    return bin01


def thinning_zhang_suen(bin01):
    thin01 = thin(bin01 > 0)
    return thin01.astype(np.uint8)


def find_endpoints(skel01):
    h, w = skel01.shape
    endpoints = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skel01[y, x] == 0:
                continue
            neighbors = np.sum(skel01[y-1:y+2, x-1:x+2]) - 1
            if neighbors == 1:
                endpoints.append((y, x))
    return endpoints


def remove_short_spurs(skel01, max_len=5):
    skel = skel01.copy()
    endpoints = find_endpoints(skel)

    for ep in endpoints:
        path = [ep]
        curr = ep
        prev = None

        for _ in range(max_len):
            y, x = curr
            neighbors = []
            for ny in range(y-1, y+2):
                for nx in range(x-1, x+2):
                    if (ny, nx) == curr:
                        continue
                    if skel[ny, nx] and (ny, nx) != prev:
                        neighbors.append((ny, nx))

            if len(neighbors) != 1:
                break

            prev = curr
            curr = neighbors[0]
            path.append(curr)

        if len(path) <= max_len:
            for y, x in path:
                skel[y, x] = 0

    return skel


def process_image(img_path, out_dir, white_th, spur_len):
    img = cv2.imread(str(img_path))
    if img is None:
        return

    name = img_path.stem
    out_img_dir = out_dir / name
    out_img_dir.mkdir(parents=True, exist_ok=True)

    bin01 = binarize_strict(img, white_th)
    binary_img = bin01 * 255

    thin01 = thinning_zhang_suen(bin01)
    thin01 = remove_short_spurs(thin01, max_len=spur_len)
    skeleton_img = thin01 * 255

    cv2.imwrite(str(out_img_dir / "binary.png"), binary_img)
    cv2.imwrite(str(out_img_dir / "skeleton.png"), skeleton_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--white_th", type=int, default=245)
    parser.add_argument("--spur_len", type=int, default=5)
    args = parser.parse_args()

    input_dir = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"}
    for img_path in input_dir.iterdir():
        if img_path.suffix.lower() in exts:
            process_image(img_path, out_dir, args.white_th, args.spur_len)


if __name__ == "__main__":
    main()
