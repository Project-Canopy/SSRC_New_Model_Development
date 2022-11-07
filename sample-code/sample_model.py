import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Softmax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy


def define_model(numclasses, input_shape):
    """
    numclasses: number of classes
    input_shape: shape of your input
    """
    
    # parameters for CNN
    input_tensor = Input(shape=input_shape, name='input')

    print(input_tensor.get_shape())

    # introduce a additional layer to get from bands to 3 input channels
    input_tensor = Conv2D(3, (1, 1))(input_tensor)

    print(input_tensor.get_shape())

    base_model_pre_trained = keras.applications.ResNet50(include_top=False,
                                                        weights='imagenet',
                                                        input_shape=(100, 100, 3))
    base_model = keras.applications.ResNet50(include_top=False,
                                                weights=None,
                                                input_tensor=input_tensor)   
    
    for i, layer in enumerate(base_model_pre_trained.layers):
        # we must skip input layer, which has no weights
        if i == 0:
            continue
        base_model.layers[i + 1].set_weights(layer.get_weights())

    # add a global spatial average pooling layer
    top_model = base_model.output
    # top_model = full_model.output
    top_model = GlobalAveragePooling2D()(top_model)

    # let's add a fully-connected layer
    top_model = Dense(2048, activation='relu')(top_model)
    top_model = Dense(2048, activation='relu')(top_model)
    # and a logistic layer
    predictions = Dense(numclasses, activation='softmax')(top_model)

    print(predictions.get_shape())

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model