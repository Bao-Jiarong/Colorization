import tensorflow as tf

class color_ae_2(tf.keras.Model):
    def __init__(self, input_shape, filters):
        super(color_ae_2, self).__init__()

        # Encoder
        self.conv1  = tf.keras.layers.Conv2D(filters, (3,3), input_shape = input_shape,  activation='relu', padding='same', strides = 2)
        self.conv2  = tf.keras.layers.Conv2D(filters * 2, (3,3), input_shape = input_shape,  activation='relu', padding='same')

        self.conv3  = tf.keras.layers.Conv2D(filters * 4, (3,3), input_shape = input_shape,  activation='relu', padding='same', strides = 2)
        self.conv4  = tf.keras.layers.Conv2D(filters * 4, (3,3), input_shape = input_shape,  activation='relu', padding='same')

        self.conv5  = tf.keras.layers.Conv2D(filters * 8, (3,3), input_shape = input_shape,  activation='relu', padding='same', strides = 2)
        self.conv6  = tf.keras.layers.Conv2D(filters * 8, (3,3), input_shape = input_shape,  activation='relu', padding='same')

        self.conv7  = tf.keras.layers.Conv2D(filters * 16, (3,3), input_shape = input_shape,  activation='relu', padding='same')

        self.upsampling1  = tf.keras.layers.UpSampling2D((2,2))
        self.conv8  = tf.keras.layers.Conv2D(filters * 16, (3,3), input_shape = input_shape,  activation='relu', padding='same')

        self.upsampling2  = tf.keras.layers.UpSampling2D((2,2))

        self.conv9  = tf.keras.layers.Conv2D(filters * 8, (3,3), input_shape = input_shape,  activation='relu', padding='same')
        self.conv10 = tf.keras.layers.Conv2D(filters * 4, (3,3), input_shape = input_shape,  activation='relu', padding='same')
        self.conv11 = tf.keras.layers.Conv2D(filters * 2, (3,3), input_shape = input_shape,  activation='relu', padding='same')

        self.conv12 = tf.keras.layers.Conv2D(3, (3,3), activation='tanh', padding='same')
        self.upsampling3  = tf.keras.layers.UpSampling2D((2,2))

    def call(self, inputs):
        x = inputs                      #;print(x.shape)
        x = self.conv1(x)               #;print(x.shape)
        x = self.conv2(x)               #;print(x.shape)
        x = self.conv3(x)               #;print(x.shape)
        x = self.conv4(x)               #;print(x.shape)
        x = self.conv5(x)               #;print(x.shape)
        x = self.conv6(x)               #;print(x.shape)
        x = self.conv7(x)               #;print(x.shape)
        x = self.upsampling1(x)         #;print(x.shape)
        x = self.conv8(x)               #;print(x.shape)
        x = self.upsampling2(x)         #;print(x.shape)
        x = self.conv9(x)               #;print(x.shape)
        x = self.conv10(x)              #;print(x.shape)
        x = self.conv11(x)              #;print(x.shape)
        x = self.conv12(x)              #;print(x.shape)
        x = self.upsampling3(x)         #;print(x.shape)
        return x
#------------------------------------------------------------------------------
def Color_AE_2(input_shape, filters):
    model = color_ae_2(input_shape=input_shape, filters=filters)
    model.build(input_shape = input_shape)
    return model
