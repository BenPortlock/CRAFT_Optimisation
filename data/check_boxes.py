import cv2
import numpy as np
from pathlib import Path

# ===== CONFIG =====
training_dir = Path("test")  # base folder
images_dir = training_dir / "images"
labels_dir = training_dir / "char_labels"
output_dir = training_dir / "debug_vis"

output_dir.mkdir(parents=True, exist_ok=True)
font_scale = 0.6
thickness = 1
box_color = (0, 255, 0)   # green
text_color = (0, 0, 255)  # red
# ==================


def order_clockwise(pts):
    """Order a quad clockwise for proper drawing."""
    pts = np.array(pts).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    return pts[np.argsort(angles)].astype(int)


def draw_quad_with_index(img, quad, idx):
    """Draw polygon and index label."""
    quad = order_clockwise(quad)
    cv2.polylines(img, [quad], isClosed=True, color=box_color, thickness=2)
    # top-left point for placing text
    tl = quad[np.argmin(quad[:, 1])]
    label_pos = (int(tl[0]), int(tl[1] - 5))
    cv2.putText(img, str(idx), label_pos,
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)


# if save_frames:
#    frame_dir.mkdir(parents=True, exist_ok=True)


def draw_one_box(img, coords, idx):
    """Draw one quad + index."""
    quad = np.array(coords, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [quad], isClosed=True, color=box_color, thickness=2)
    # text at top-left
    tl_x, tl_y = coords[0], coords[1]
    cv2.putText(img, f"Row {idx}", (int(tl_x), int(tl_y)-5),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)


image_paths = sorted(images_dir.glob("*"))
image_idx = 0

while 0 <= image_idx < len(image_paths):
    img_path = image_paths[image_idx]
    label_path = labels_dir / (img_path.stem + ".txt")
    if not label_path.exists():
        print(f"⚠ No label file for {img_path.name}")
        image_idx += 1
        continue

    img_orig = cv2.imread(str(img_path))
    if img_orig is None:
        print(f"⚠ Could not read {img_path}")
        image_idx += 1
        continue

    # Load boxes
    boxes = []
    with open(label_path, "r") as f:
        for idx, line in enumerate(f.readlines()):
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            coords = np.array(list(map(float, parts[-8:]))).reshape(4, 2)
            if np.all(coords >= 0) and np.all(coords <= 1):
                coords[:, 0] *= img_orig.shape[1]
                coords[:, 1] *= img_orig.shape[0]
            boxes.append((coords.flatten(), idx))

    if not boxes:
        image_idx += 1
        continue

    box_idx = 0
    while True:
        img = img_orig.copy()
        draw_one_box(img, boxes[box_idx][0], boxes[box_idx][1])
        cv2.imshow("Bounding Box Viewer", img)
        key = cv2.waitKey(0)

        if key == ord('q'):      # quit everything
            image_idx = len(image_paths)
            break
        elif key == ord('n'):    # next box
            box_idx = (box_idx + 1) % len(boxes)
        elif key == ord('p'):    # previous box
            box_idx = (box_idx - 1) % len(boxes)
        elif key == ord('>'):    # next image
            image_idx += 1
            break
        elif key == ord('<'):    # previous image
            image_idx = max(0, image_idx - 1)
            break

cv2.destroyAllWindows()
