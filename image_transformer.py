import tensorflow as tf
import numpy as np
import cv2


class ImageTransformer:
    def __init__(self, origin_shape=(512, 288, 3), out_shape=(84, 84), crop_boundaries=(0, 50, 400, 238)):
        with tf.variable_scope("image_transformer"):
            self.frame_height = out_shape[0]
            self.frame_width = out_shape[1]
            self.input_img = tf.placeholder(shape=origin_shape, dtype=tf.uint8)

            self.output = tf.image.rgb_to_grayscale(self.input_img)
            self.output = tf.image.crop_to_bounding_box(self.output, *crop_boundaries)
            self.output = tf.image.resize_images(self.output, size=out_shape,
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def transform(self, image, sess=None):
        # gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # cropped = gray[self.crop_boundaries[0]:self.crop_boundaries[2], self.crop_boundaries[1]:self.crop_boundaries[3]]
        # resized = cv2.resize(cropped, self.out_shape)
        # thresholded = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        # normalized = np.zeros_like(thresholded)
        # normalized = cv2.normalize(thresholded, normalized, 0, 1, cv2.NORM_MINMAX)
        # thresholded = cv2.threshold(resized, 100, 155, cv2.THRESH_BINARY)[1]
        return sess.run(self.output, feed_dict={self.input_img: image})
        # return sess.run(self.converted, feed_dict={self.converted: normalized})
