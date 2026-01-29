# nst.py

import tensorflow as tf
import numpy as np
import PIL.Image


class NeuralStyleTransfer:
    def __init__(self,
                 content_layers=None,
                 style_layers=None,
                 style_weight=1e-2,
                 content_weight=1e4,
                 max_dim=512):

        if content_layers is None:
            content_layers = ['block5_conv2']
        if style_layers is None:
            style_layers = ['block1_conv1',
                            'block2_conv1',
                            'block3_conv1',
                            'block4_conv1',
                            'block5_conv1']

        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)

        self.style_weight = style_weight
        self.content_weight = content_weight
        self.max_dim = max_dim

        self.vgg = self._vgg_layers(self.style_layers + self.content_layers)
        self.vgg.trainable = False

    def _vgg_layers(self, layer_names):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        outputs = [vgg.get_layer(name).output for name in layer_names]
        return tf.keras.Model([vgg.input], outputs)

    def load_img(self, path_to_img):
        img = tf.io.read_file(path_to_img)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        scale = self.max_dim / max(shape)
        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        return img[tf.newaxis, :]

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def style_content_loss(self, outputs, style_targets, content_targets):
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        style_loss = tf.add_n([
            tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
            for name in style_outputs
        ])
        style_loss *= self.style_weight / self.num_style_layers

        content_loss = tf.add_n([
            tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
            for name in content_outputs
        ])
        content_loss *= self.content_weight / self.num_content_layers

        return style_loss + content_loss

    def run(self, content_path, style_path,
            epochs, steps_per_epoch):

        content_image = self.load_img(content_path)
        style_image = self.load_img(style_path)

        extractor = StyleContentModel(
            self.vgg,
            self.style_layers,
            self.content_layers,
            self.gram_matrix
        )

        style_targets = extractor(style_image)['style']
        content_targets = extractor(content_image)['content']

        image = tf.Variable(content_image)
        opt = tf.optimizers.Adam(learning_rate=0.02)

        losses = []

        @tf.function()
        def train_step(image):
            with tf.GradientTape() as tape:
                outputs = extractor(image)
                loss = self.style_content_loss(outputs,
                                               style_targets,
                                               content_targets)

            grad = tape.gradient(loss, image)
            opt.apply_gradients([(grad, image)])
            image.assign(tf.clip_by_value(image, 0.0, 1.0))
            return loss

        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                loss = train_step(image)
                losses.append(loss.numpy())

        return image, losses


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, vgg, style_layers, content_layers, gram_matrix):
        super().__init__()
        self.vgg = vgg
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.gram_matrix = gram_matrix

    def call(self, inputs):
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)

        style_outputs = outputs[:self.num_style_layers]
        content_outputs = outputs[self.num_style_layers:]

        style_outputs = [self.gram_matrix(o) for o in style_outputs]

        return {
            'content': dict(zip(self.content_layers, content_outputs)),
            'style': dict(zip(self.style_layers, style_outputs))
        }
