##################################################################
### author: Matthew Davis                                      ###
### CSC4040 - Computer Vision - Fall 2025                      ###
### Final Project: Face Blurring Pipeline                      ###
##################################################################

##################################################################
### Image Sources                                              ###
### https://www.kaggle.com/datasets/ngoduy/dataset-for-face-detection
### https://www.freepik.com/search?format=search&last_filter=people_range&last_value=4&people=include&people_range=4&query=People&selection=1&type=photo#uuid=89a047c3-bbb8-43f7-8a17-dbb42f6549ba
### https://cocodataset.org/#explore
###################################################################

import os
import cv2

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}

# Prompting functions -------------------------------------------------------
def prompt_dir(prompt_text: str) -> str:
    while True:
        p = input(prompt_text).strip().strip('"').strip("'")
        p = os.path.expanduser(p)
        if os.path.isdir(p):
            return p
        print(f"Directory not found: {p}")


def prompt_file(prompt_text: str) -> str:
    while True:
        p = input(prompt_text).strip().strip('"').strip("'")
        p = os.path.expanduser(p)
        if os.path.isfile(p):
            return p
        print(f"File not found: {p}")


def prompt_float(prompt_text: str, default: float) -> float:
    s = input(prompt_text).strip()
    if not s:
        return default
    try:
        v = float(s)
        return v
    except ValueError:
        print(f"Invalid number. Using default {default}.")
        return default


def prompt_choice(prompt_text: str, choices: set[str], default: str) -> str:
    s = input(prompt_text).strip().lower()
    if not s:
        return default
    if s in choices:
        return s
    print(f"Invalid choice. Using default '{default}'.")
    return default

# End prompting functions ---------------------------------------------------

# Utility functions ---------------------------------------------------------
def list_images(folder: str) -> list[str]:
    files = []
    for name in os.listdir(folder):
        full = os.path.join(folder, name)
        if os.path.isfile(full):
            _, ext = os.path.splitext(name)
            if ext.lower() in SUPPORTED_EXTS:
                files.append(full)
    return sorted(files)


def ensure_output_folders(base_out: str) -> tuple[str, str]:
    blurred = os.path.join(base_out, "Blurred")
    nofaces = os.path.join(base_out, "NoFaces")
    os.makedirs(blurred, exist_ok=True)
    os.makedirs(nofaces, exist_ok=True)
    return blurred, nofaces

# Face detection functions -----------------------------------------------------

# Load DNN face detector model
def load_face_net(pb_path: str, pbtxt_path: str) -> cv2.dnn_Net:
    # OpenCV DNN face detector model
    return cv2.dnn.readNetFromTensorflow(pb_path, pbtxt_path)

# Detect faces using DNN model
def detect_faces_dnn(net: cv2.dnn_Net, image_bgr, conf_threshold: float):
    """
    Returns a list of face boxes as (x1, y1, x2, y2, conf).
    Model output format (typical): [1, 1, N, 7]
      [image_id, class_id, confidence, x1, y1, x2, y2] with coords normalized [0..1].
    """
    h, w = image_bgr.shape[:2]

    # Prepare input blob and perform forward pass
    blob = cv2.dnn.blobFromImage(
        image_bgr,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False
    )
    net.setInput(blob)
    out = net.forward()

    # Parse detections
    boxes = []
    for i in range(out.shape[2]):
        conf = float(out[0, 0, i, 2])
        if conf < conf_threshold:
            continue

        x1 = int(out[0, 0, i, 3] * w)
        y1 = int(out[0, 0, i, 4] * h)
        x2 = int(out[0, 0, i, 5] * w)
        y2 = int(out[0, 0, i, 6] * h)

        # reorder + clamp
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2, conf))

    return boxes

# Face blurring functions -----------------------------------------------------

# Determine if image is low-res based on max side length
def is_low_res(img_bgr, max_side=800):
    h, w = img_bgr.shape[:2]
    return max(h, w) <= max_side

# Smart Gaussian blur based on image resolution
def blur_roi_gaussian_smart(img_bgr, x1, y1, x2, y2):
    roi = img_bgr[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 2 or rw < 2:
        return

    # Decide blur method based on image resolution
    if is_low_res(img_bgr):
        # Low-res: kernel-based Gaussian proportional to face size
        k = int(min(rw, rh) * 0.6)  # strong relative blur
        k = max(15, k)
        if k % 2 == 0:
            k += 1
        img_bgr[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
    else:
        # High-res: downscale -> heavy blur -> upscale (very strong anonymization)
        scale = 0.06
        small_w = max(2, int(rw * scale))
        small_h = max(2, int(rh * scale))

        small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_AREA)
        k = max(3, (min(small_w, small_h) // 2) * 2 + 1)
        small_blurred = cv2.GaussianBlur(small, (k, k), 0)
        strong = cv2.resize(small_blurred, (rw, rh), interpolation=cv2.INTER_LINEAR)
        img_bgr[y1:y2, x1:x2] = strong


# Smart pixelation based on image resolution
def blur_roi_pixelate_smart(img_bgr, x1, y1, x2, y2):
    roi = img_bgr[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh < 2 or rw < 2:
        return

    # Decide pixelation grid size based on image resolution
    if is_low_res(img_bgr):
        grid = 10   # low-res faces need a small grid to be obvious
    else:
        grid = 12   # high-res already looks strong at 12 (or use 10 if you want extreme)

    small = cv2.resize(roi, (grid, grid), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    img_bgr[y1:y2, x1:x2] = pixelated

# Apply face blurring to all detected boxes
def apply_face_blur(img_bgr, boxes, mode: str):
    """
    mode: 'gaussian' or 'pixelate'
    """
    # define image dimensions here
    h, w = img_bgr.shape[:2]

    for (x1, y1, x2, y2, conf) in boxes:
        # Pad by 25% of face size
        pad = int(min(x2 - x1, y2 - y1) * 0.25)

        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(w, x2 + pad)   # slicing end is exclusive
        y2p = min(h, y2 + pad)

        if mode == "gaussian":
            blur_roi_gaussian_smart(img_bgr, x1p, y1p, x2p, y2p)
        else:
            blur_roi_pixelate_smart(img_bgr, x1p, y1p, x2p, y2p)

def main():
    print("\n=== Final Project: Face Blurring Pipeline ===\n")

    in_dir = prompt_dir("Enter path to your input images folder (30 images): ")
    out_dir = prompt_dir("Enter path to output folder: ")

    pb_path = prompt_file("Enter path to opencv_face_detector_uint8.pb: ")
    pbtxt_path = prompt_file("Enter path to opencv_face_detector.pbtxt: ")

    conf_threshold = prompt_float("Enter confidence threshold (default 0.50): ", 0.50)
    blur_mode = prompt_choice(
        "Choose blur mode ('gaussian' or 'pixelate', default gaussian): ",
        {"gaussian", "pixelate"},
        "gaussian"
    )

    blurred_dir, nofaces_dir = ensure_output_folders(out_dir)
    images = list_images(in_dir)

    print(f"\nFound {len(images)} image file(s) to process.")
    print(f"Threshold: {conf_threshold}")
    print(f"Blur mode: {blur_mode}")
    print(f"Output -> {out_dir}")
    print("Subfolders: Blurred/ and NoFaces/\n")

    net = load_face_net(pb_path, pbtxt_path)

    processed = 0
    images_with_faces = 0
    total_faces = 0

    # Process each image
    for img_path in images:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"WARNING: Could not read {img_path}, skipping.")
            continue

        boxes = detect_faces_dnn(net, img, conf_threshold)
        filename = os.path.basename(img_path)

        if len(boxes) > 0:
            total_faces += len(boxes)
            images_with_faces += 1
            # blur in-place
            apply_face_blur(img, boxes, blur_mode)
            cv2.imwrite(os.path.join(blurred_dir, filename), img)
            print(f"BLURRED: {filename} (faces: {len(boxes)})")
        else:
            cv2.imwrite(os.path.join(nofaces_dir, filename), img)
            print(f"NO FACES: {filename}")

        processed += 1

    print("\n=== Summary ===")
    print(f"Images processed: {processed}")
    print(f"Images with faces detected: {images_with_faces}")
    print(f"Total faces detected (all images): {total_faces}")
    print(f"Blurred outputs saved to: {blurred_dir}")
    print(f"No-face outputs saved to: {nofaces_dir}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()