#Custom L1 Distance Layer will come here

#Import Dependencies
from keras.layers import Layer
import tensorflow as tf
#Custom L1 Distance Layer from Jupyter
#We need to load our model using custom objects and we made our distance layer a custom object
class L1Dist(Layer):
    
    # Init method - inheritance
    def __init__(self, **kwargs):
        super().__init__()
       
    # Magic happens here - similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    



