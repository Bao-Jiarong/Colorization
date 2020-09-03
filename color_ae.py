'''
  Author       : Bao Jiarong
  Creation Date: 2020-09-01
  email        : bao.salirong@gmail.com
  Task         : Colorization_AE
'''
import tensorflow as tf

class color_ae(tf.keras.Model):
    def __init__(self, filters = 32):
        super(color_ae, self).__init__()

        # Encoder
        self.conv1  = tf.keras.layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')
        self.conv2  = tf.keras.layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same', strides=2)
        self.conv3  = tf.keras.layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')
        self.conv4  = tf.keras.layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same', strides=2)
        self.conv5  = tf.keras.layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')
        self.conv6  = tf.keras.layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same', strides=2)

        # Latent
        self.conv7  = tf.keras.layers.Conv2D(filters*16, (3, 3), activation='relu', padding='same')

        # Decoder
        self.conv8  = tf.keras.layers.Conv2D(filters*8, (3, 3), activation='relu', padding='same')
        self.conv9  = tf.keras.layers.Conv2D(filters*4, (3, 3), activation='relu', padding='same')
        self.samp1  = tf.keras.layers.UpSampling2D((2, 2))
        self.conv10 = tf.keras.layers.Conv2D(filters*2, (3, 3), activation='relu', padding='same')
        self.samp2  = tf.keras.layers.UpSampling2D((2, 2))
        self.conv11 = tf.keras.layers.Conv2D(filters, (3, 3), activation='relu', padding='same')
        self.conv12 = tf.keras.layers.Conv2D(3, (3, 3), activation='tanh', padding='same')
        self.samp3  = tf.keras.layers.UpSampling2D((2, 2))

    def call(self, inputs):
        x = inputs
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.samp1(x)
        x = self.conv10(x)
        x = self.samp2(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.samp3(x)
        return x

#------------------------------------------------------------------------------
def Color_AE(input_shape, filters):
    model = color_ae(filters = filters)
    model.build(input_shape = input_shape)
    return model
