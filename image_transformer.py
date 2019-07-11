import tensorflow as tf


class ImageTransformer:
    def __init__(self, origin_shape=(512, 288, 3), out_shape=(84, 84), crop_boundaries=(0, 50, 400, 238)):
        with tf.variable_scope("image_transformer"):
            self.input_img = tf.placeholder(shape=origin_shape, dtype=tf.int8)
            self.output = tf.image.rgb_to_grayscale(self.input_img)
            self.output = tf.image.crop_to_bounding_box(self.output, *crop_boundaries)
            self.output = tf.image.resize_images(self.output, size=out_shape,
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def transform(self, image, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.output, feed_dict={self.input_img: image})
