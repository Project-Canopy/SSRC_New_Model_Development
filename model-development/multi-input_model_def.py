import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Softmax
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.backend import categorical_crossentropy
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import layers
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy

def define_model(numclasses, input_shape,pre_trained_model="resnet50"):
        #MERGING INPUTS AT DEPTH 6, AS ACCORDING TO https://www.azavea.com/blog/2019/08/30/transfer-learning-from-rgb-to-multi-band-imagery/
        #This corresponds to merging the outputs after conv2_block3_out

        print(f"Using Pre-trained {pre_trained_model} model")
        
        # parameters for CNN
        input_tensor = Input(shape=input_shape)

        # introduce a additional layer to get from bands to 3 input channels
        input_tensor = Conv2D(3, (1, 1))(input_tensor)      #TODO: CHECK THIS, Why (1,1)
        
        if pre_trained_model == "resnet50":
            RGB_base_model = keras.applications.ResNet50(include_top=False,
                                                              weights='imagenet',
                                                              input_shape=(100, 100, 3))
            
            xbands_base_model = keras.applications.ResNet50(include_top=False,
                                                     weights=None,
                                                     input_tensor=input_tensor)  #Ideally, this would be a resnet50 trained on the extra bands

            for layer in xbands_base_model.layers :
                layer._name = layer.name + str('_x')

            premerge_RGB_model =  Model(inputs=RGB_base_model.input, outputs=RGB_base_model.get_layer('conv2_block3_out').output)
            premerge_xbands_model = Model(inputs=xbands_base_model.input, outputs=xbands_base_model.get_layer('conv2_block3_out_x').output)
            merged_features = Add()([premerge_RGB_model.output, premerge_xbands_model.output])
            premerge_model = Model(inputs= [premerge_RGB_model.input, premerge_xbands_model.input],outputs = merged_features)
            postmerge_model = Model(inputs = xbands_base_model.get_layer('conv3_block1_1_conv_x').input, outputs= xbands_base_model.output)
            full_output = postmerge_model(premerge_model.output)
            full_model = Model(inputs= [premerge_RGB_model.input, premerge_xbands_model.input], outputs= full_output)

            return full_model

if __name__ == "__main__":
    
    model = define_model(5,(100,100,18))
    pdb.set_trace()