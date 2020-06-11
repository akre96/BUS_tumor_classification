
import tensorflow as tf
import numpy as np
def generator_minimax_loss(predicted_fake,  
                           nonsaturating = True, 
                           print_loss = True,
                           label_smoothing = 0):
    #                       reduction = tf.keras.losses.Reduction.SUM):
    """why we need nonsaturating?
    
       We want to increase the generator's stability when 
       it's stuck by increasing "confidence" of fake by taking the reciprocal 
       of loss function
    """
    crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits= True, label_smoothing = label_smoothing)

    if nonsaturating:
        target = tf.ones_like(predicted_fake)
        G_loss = crossentropy(target, predicted_fake)    
    else:
        target = tf.zeros_like(fake)
        G_loss = -1 * crossentropy(target,predicted_fake)
    if print_loss:
	    print("Generator_minimax_loss: ", tf.reduce_mean(G_loss))
    return G_loss


def cross_entropy_generator_loss():
	return True

def wasserstein_generator_loss():
	return True