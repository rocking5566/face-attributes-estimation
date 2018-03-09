import os
import cv2
import numpy as np
import argparse
import time
from age_gender_model import age_gender_baseline_net


# for face detection
face_detection_model_path = './model/lbpcascade_frontalface_improved.xml'
age_gender_detection_model_path = './model/age_gender_baseline.hdf5'


def load_model(face_cascade_model_path, age_gender_baseline_model_path):
    face_cascade = cv2.CascadeClassifier(face_cascade_model_path)
    age_gender_baseline_image_shape0 = 64
    age_gender_baseline_model = \
        age_gender_baseline_net((age_gender_baseline_image_shape0, age_gender_baseline_image_shape0, 3))
    age_gender_baseline_model.load_weights(age_gender_baseline_model_path)
    return face_cascade, age_gender_baseline_model, age_gender_baseline_image_shape0


def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
               font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def get_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(gray, 1.1)


def main():
    face_cascade, age_baseline_model, age_model_shape = \
        load_model(face_detection_model_path, age_gender_detection_model_path)

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print('No video camera found')
        exit()

    while True:
        ret, frame = video_capture.read()
        # tic = time.process_time()
        face_rois = get_faces(frame, face_cascade)
        # toc = time.process_time()
        # print("----- Face detection time = " + str(1000 * (toc - tic)) + "ms")

        x_test = np.empty((len(face_rois), age_model_shape, age_model_shape, 3), dtype=np.uint8)
        frame_h, frame_w, _ = np.shape(frame)

        for i, (x, y, w, h) in enumerate(face_rois):
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            xw1 = max(int(x1 - 0.9 * w), 0)
            yw1 = max(int(y1 - 0.9 * h), 0)
            xw2 = min(int(x2 + 0.9 * w), frame_w - 1)
            yw2 = min(int(y2 + 0.9 * h), frame_h - 1)

            x_test[i, :, :, :] = cv2.resize(frame[yw1:yw2 + 1, xw1:xw2 + 1, :], (age_model_shape, age_model_shape))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (xw1, yw1), (xw2, yw2), (0, 0, 255), 2)

        if len(face_rois) > 0:
            # tic = time.process_time()
            [pred_age, pred_gender] = age_baseline_model.predict(x_test)
            # toc = time.process_time()
            # print("----- Age detection time = " + str(1000 * (toc - tic)) + "ms")

            for i, (x, y, w, h) in enumerate(face_rois):
                gender = 'Male' if pred_gender[i] < 0.3 else 'Female'
                label = "{} {}".format(int(pred_age[i]), gender)
                draw_label(frame, (x, y), label)

        cv2.imshow('Video', frame)

        # Press q to exit the program
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
