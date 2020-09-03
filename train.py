'''
  Author       : Bao Jiarong
  Creation Date: 2020-09-01
  email        : bao.salirong@gmail.com
  Task         : Colorization_AE
  Dataset      : anime dataset
'''
import os
import sys
import cv2
import loader
import random
import color_ae
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# np.random.seed(7)
# tf.random.set_seed(7)
# np.set_printoptions(threshold=np.inf)

# The following part is to use gpu if you have it.
# -----------------------------------------------
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
# -----------------------------------------------

# Input/Ouptut Parameters
width      = 224
height     = 224
channel    = 3
model_name = "models/anime/checkpoint"
data_path  = "../data_img/colorization/"

# Step 0: Global Parameters
epochs     = 300
lr_rate    = 0.00001
batch_size = 8

# Step 1: Create Model
model = color_ae.Color_AE((None,height, width, channel), filters = 32)

if sys.argv[1] == "train":
    print(model.summary())

    # Step 2: Load weights:
    try:
        model.load_weights(model_name)
        print("weights were successfully loaded")
    except:
        print("Weights were not loaded")

    # Step 3: Load data
    X_train, Y_train, X_valid, Y_valid = loader.load_light(data_path,width,height,True,0.8,False)
    # X_train, Y_train, X_valid, Y_valid = loader.load_heavy(data_path,width,height,False,0.8,False)

    # Step 4: Define The Optimizer and the loss
    optimizer= tf.keras.optimizers.Adam(learning_rate=lr_rate)

    # Define The Loss
    #---------------------
    @tf.function
    def my_loss(y_true, y_pred):
        return tf.keras.losses.MSE(y_true=y_true, y_pred=y_pred)

    # Define The Metrics
    tr_loss = tf.keras.metrics.MeanSquaredError(name = 'tr_loss')
    va_loss = tf.keras.metrics.MeanSquaredError(name = 'va_loss')

    #---------------------
    @tf.function
    def train_step(X, Y_true):
        with tf.GradientTape() as tape:
            Y_pred = model(X, training=True)
            loss   = my_loss(y_true=Y_true, y_pred=Y_pred )
        gradients= tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        tr_loss.update_state(y_true = Y_true, y_pred = Y_pred )

    #---------------------
    @tf.function
    def valid_step(X, Y_true):
        Y_pred= model(X, training=False)
        loss  = my_loss(y_true=Y_true, y_pred=Y_pred)
        va_loss.update_state(y_true = Y_true, y_pred = Y_pred)

    #---------------------
    # Step 5: start training
    L = len(X_train)
    M = len(X_valid)
    tr_steps = int(L/batch_size)
    va_steps = int(M/batch_size)

    for epoch in range(epochs):
        # Step 6: Run on training data + Update weights
        for step in range(tr_steps):
            sketch, colored = loader.get_batch_light(X_train, Y_train, batch_size, width, height, True)
            # sketch, colored = loader.get_batch_heavy(X_train, Y_train, batch_size)
            train_step(sketch,colored)

            print(epoch,"/",epochs,step,tr_steps,"loss:",tr_loss.result().numpy(),end="\r")

        # Step 7: Run on validation data without updating weights
        for step in range(va_steps):
            sketch, colored = loader.get_batch_light(X_valid, Y_valid, batch_size, width, height)
            # sketch, colored = loader.get_batch_heavy(X_valid, Y_valid, batch_size)
            valid_step(sketch, colored)

        print(epoch,"/",epochs,
              "tr_loss:",tr_loss.result().numpy(),
              "va_loss:",va_loss.result().numpy())

        # Step 8: Save the model for each epoch
        model.save_weights(filepath=model_name, save_format='tf')

elif sys.argv[1] == "predict":
    # Step 1: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 2: Prepare the input
    img_name= sys.argv[2]
    img     = cv2.imread(img_name)
    w       = img.shape[1] >> 1
    sketch  = cv2.resize(img[:,w:,:],(height,width),interpolation = cv2.INTER_AREA)
    gr_true = cv2.resize(img[:,:w,:],(height,width),interpolation = cv2.INTER_AREA)
    images  = np.array([sketch])
    images  = loader.scaling_tech(images,method="normalization")
    gr_true = loader.scaling_tech(gr_true,method="normalization")

    # Step 3: Predict the class
    preds  = my_model.predict(images)
    preds  = (preds[0] - preds.min())/(preds.max() - preds.min())
    images = np.hstack((images[0],preds,gr_true))
    cv2.imshow("imgs",images)
    cv2.waitKey(0)

elif sys.argv[1] == "predict_mine":
    # Step 1: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 2: Prepare the input
    img_name= sys.argv[2]
    img     = cv2.imread(img_name)
    sketch  = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
    gr_true = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
    images  = np.array([sketch])
    images  = loader.scaling_tech(images, method="normalization")
    gr_true = loader.scaling_tech(gr_true,method="normalization")

    # Step 3: Predict the class
    preds   = my_model.predict(images)
    preds   = (preds[0] - preds.min())/(preds.max() - preds.min())
    images  = np.hstack((images[0],preds,gr_true))
    cv2.imshow("imgs",images)
    cv2.waitKey(0)

elif sys.argv[1] == "predict_all":
    # Step 1: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 2: Prepare the input
    imgs_filenames = sorted([os.path.join("../data_img/colorization/test/", file)
                             for file in os.listdir("../data_img/colorization/test/")],
                             key=os.path.getctime)
    L = len(imgs_filenames)
    i = np.random.randint(0,L - 11)
    imgs_filenames = imgs_filenames[i:i+10]
    images  = []
    for filename in imgs_filenames:
        img = cv2.imread(filename)
        w   = img.shape[1] >> 1
        sketch= img[:,w:,:]
        image = cv2.resize(sketch,(height,width),interpolation = cv2.INTER_AREA)
        images.append(image)

    # True images
    images = np.array(images)
    images = loader.scaling_tech(images,method="normalization")

    # Step 3: Predicted images
    preds = my_model.predict(images)
    true_images = np.hstack(images)
    pred_images = np.hstack(preds)

    images = np.vstack((true_images, pred_images))
    h = images.shape[0]
    w = images.shape[1]
    images = cv2.resize(images,(w << 0, h << 0))
    cv2.imshow("imgs",images)
    cv2.waitKey(0)
