import os
from options.test_options import TestOptions
from mtcnn import MTCNN
import cv2


def get_data_path(root='custom_images'):
    im_path = [os.path.join(root, i) for i in sorted(
        os.listdir(root)) if i.endswith('png') or i.endswith('jpg')]
    lm_path = [i.replace('png', 'txt').replace('jpg', 'txt') for i in im_path]
    lm_path = [os.path.join(i.replace(i.split(
        os.path.sep)[-1], ''), 'detections', i.split(os.path.sep)[-1]) for i in lm_path]

    return im_path, lm_path


def draw_landmark(source_image, keypoint, image_path, image_index):
  # Draw the keypoint.
  cv2.circle(source_image, keypoint, radius=8,
             color=(0, 0, 255), thickness=-1)
  cv2.imshow('Image #' + str(image_index) + ' - ' +
             image_path[image_index], source_image)


def main(opt, name='custom_images'):
  im_path, lm_path = get_data_path(name)

  for i in range(len(im_path)):
    print('Detect landmarks:', i, im_path[i])

    # source_image = cv2.cvtColor(cv2.imread(im_path[i]), cv2.COLOR_BGR2RGB)
    source_image = cv2.imread(im_path[i])

    # Launch faces detector only if there is no existing landmarks file for
    # this image.
    if os.path.isfile(lm_path[i]):
      # Get the keypoints from the landmark file.
      with open(lm_path[i], 'r') as landmarks_file:
        lines = landmarks_file.readlines()
        landmarks_file.close()

        for line in lines:
          # Get the keypoint from each line and convert it in a floats tuple.
          keypointsStrList = line.strip().split('\t')

          # `cv2.circle()` expects a tuple of integers as the `center`
          # parameter, so we need to remove the float part of each
          # coordinate string.
          keypoint = (int(keypointsStrList[0].split('.')[0]),
                      int(keypointsStrList[1].split('.')[0]))

          # Only for debugging purposes.
          print('Keypoint type:' + str(type(keypoint)))
          for coordinate in keypoint:
            print('Keypoint coordinate value:' +
                  str(coordinate) + '; type:' + str(type(coordinate)))

          draw_landmark(source_image, keypoint, im_path, i)
    else:
      detector = MTCNN()
      detected_faces = detector.detect_faces(source_image)
      print('Detected Faces: ', detected_faces)

      keypoints = detected_faces[0]['keypoints'].values()
      landmarks = ''

      for keypoint in keypoints:
        # Append the current keypoint to the output string `landmarks`.
        landmarks += str(keypoint[0]) + '.0\t' + str(keypoint[1]) + '.0\n'

        draw_landmark(source_image, keypoint, im_path, i)

        with open(lm_path[i], 'w') as landmarks_file:
          landmarks_file.write(landmarks)
          landmarks_file.close()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
  opt = TestOptions().parse()
  main(opt, opt.img_folder)
