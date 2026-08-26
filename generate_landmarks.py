from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def yunet_to_deep3d_landmarks(
    detection: np.ndarray,
) -> np.ndarray:
    """Extract Deep3DFace-compatible 5-point landmarks from YuNet.

    YuNet detection layout:
        0:4   bounding box: x, y, width, height
        4:6   right eye
        6:8   left eye
        8:10  nose tip
        10:12 right mouth corner
        12:14 left mouth corner
        14    confidence

    This five-point ordering matches Deep3DFace preprocessing.
    """
    detection = np.asarray(detection, dtype=np.float32)

    if detection.shape[0] < 15:
        raise ValueError(
            f"Expected YuNet detection with at least 15 values, "
            f"got shape {detection.shape}"
        )

    return detection[4:14].reshape(5, 2)


def create_detector(
    model_path: Path,
    score_threshold: float,
) -> cv2.FaceDetectorYN:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YuNet model not found: {model_path}"
        )

    return cv2.FaceDetectorYN.create(
        str(model_path),
        "",
        (320, 320),
        score_threshold,
        0.3,
        5000,
    )


def detect_face(
    detector: cv2.FaceDetectorYN,
    image: np.ndarray,
) -> np.ndarray | None:
    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        return None

    # YuNet confidence is the final value in each detection.
    best_index = int(np.argmax(faces[:, -1]))

    return faces[best_index]


def process_image(
    detector: cv2.FaceDetectorYN,
    image_path: Path,
    detection_dir: Path,
) -> bool:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] Could not read {image_path}")
        return False

    face = detect_face(detector, image)

    if face is None:
        print(f"[NO FACE] {image_path.name}")
        return False

    landmarks = yunet_to_deep3d_landmarks(face)

    output_path = detection_dir / f"{image_path.stem}.txt"

    np.savetxt(
        output_path,
        landmarks,
        fmt="%.6f",
    )

    confidence = float(face[-1])

    print(
        f"[OK] {image_path.name} "
        f"-> {output_path} "
        f"(confidence={confidence:.4f})"
    )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Deep3DFaceRecon-compatible five-point "
            "landmark files using OpenCV YuNet."
        )
    )

    parser.add_argument(
        "--img_folder",
        required=True,
        type=Path,
        help="Folder containing input JPG/PNG images.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "assets/yunet/"
            "face_detection_yunet_2023mar.onnx"
        ),
        help="Path to the YuNet ONNX model.",
    )

    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.9,
        help="Minimum YuNet face-detection confidence.",
    )

    args = parser.parse_args()

    img_folder = args.img_folder

    if not img_folder.is_dir():
        raise NotADirectoryError(
            f"Image folder does not exist: {img_folder}"
        )

    detection_dir = img_folder / "detections"
    detection_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in img_folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    )

    if not image_paths:
        raise RuntimeError(
            f"No JPG/PNG images found in {img_folder}"
        )

    detector = create_detector(
        args.model,
        args.score_threshold,
    )

    successful = 0

    for image_path in image_paths:
        if process_image(
            detector,
            image_path,
            detection_dir,
        ):
            successful += 1

    print()
    print(
        f"Generated landmarks for "
        f"{successful}/{len(image_paths)} images."
    )


if __name__ == "__main__":
    main()

