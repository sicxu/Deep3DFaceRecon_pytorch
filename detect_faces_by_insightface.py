# coding=utf-8
"""
Detect faces and landmarks in images using the InsightFace library.

Authors:
    zhaoyafei (zhaoyafei0210@gmail.com, https://github.com/walkoncross)
"""

import os
import os.path as osp

import cv2
from insightface.app import FaceAnalysis


def detect_faces(input_dir):
    # Initialize the face analysis app
    app = FaceAnalysis(
        allowed_modules=["detection", "alignment"],
        providers=[
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Create the output directory if it doesn't exist
    output_dir = os.path.join(input_dir, "detections")
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all images in the input directory
    file_list = [
        ff
        for ff in os.listdir(input_dir)
        if osp.splitext(ff)[-1].lower() in [".png", ".jpg", ".jpeg", ".bmp"]
    ]

    print(f"--> image files: \n{file_list}")

    for ii, filename in enumerate(file_list):
        print(f"--> {ii}: {filename}")
        image_path = os.path.join(input_dir, filename)
        img = cv2.imread(image_path)

        # Detect faces in the image
        faces = app.get(img)

        if faces:
            print(f"Detected {len(faces)} faces")
            # Get the face with the highest score
            best_face = max(faces, key=lambda face: face["det_score"])

            # Get the 5 key points of the best face
            keypoints = best_face["kps"]

            # Write the key points to a txt file
            output_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}.txt"
            )
            with open(output_path, "w") as f:
                for point in keypoints:
                    f.write(f"{point[0]} {point[1]}\n")
        else:
            print(f"No faces detected")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing images")
    args = parser.parse_args()

    detect_faces(args.input_dir)
