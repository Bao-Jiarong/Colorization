import tensorflow as tf

# class Block(tf.keras.layers.Layer):
#     def __init__(self, n, filters, activation="relu", strides= (1,1)):
#         super(Block, self).__init__()
#         for i in range(n):
#         self.conv = tf.keras.layers.Conv2D(filters = filters, kernel_size = (3,3), activation  = activation, padding = 'same', strides = strides)
#
#     def call(self, inputs, **kwargs):
#         x = inputs
#         x = self.conv(x)
#         x = self.pool(x)
#         return x

class color_ae_1(tf.keras.Model):
    def __init__(self):
        super(color_ae_1, self).__init__()

        # Encoder
        self.conv1  = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
        self.conv2  = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')
        self.conv3  = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.conv4  = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2)
        self.conv5  = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.conv6  = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', strides=2)

        self.upsampling1  = tf.keras.layers.UpSampling2D((2, 2))
        self.conv7  = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')
        self.upsampling2  = tf.keras.layers.UpSampling2D((2, 2))
        self.conv8  = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')
        self.upsampling3  = tf.keras.layers.UpSampling2D((2, 2))
        self.conv9  = tf.keras.layers.Conv2D(3, (3, 3), activation='tanh', padding='same')

    def call(self, inputs):
        x = inputs                      #;print(x.shape)
        x = self.conv1(x)               #;print(x.shape)
        x = self.conv2(x)               #;print(x.shape)
        x = self.conv3(x)               #;print(x.shape)
        x = self.conv4(x)               #;print(x.shape)
        x = self.conv5(x)               #;print(x.shape)
        x = self.conv6(x)               #;print(x.shape)
        x = self.upsampling1(x)         #;print(x.shape)
        x = self.conv7(x)              #;print(x.shape)
        x = self.upsampling2(x)         #;print(x.shape)
        x = self.conv8(x)              #;print(x.shape)
        x = self.upsampling3(x)         #;print(x.shape)
        x = self.conv9(x)              #;print(x.shape)
        return x

#------------------------------------------------------------------------------
def Color_AE_1(input_shape):
    model = color_ae_1()
    model.build(input_shape = input_shape)
    return model
