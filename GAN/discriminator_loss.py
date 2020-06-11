import tensorflow as tf
import numpy as np
"""try to implement an entropy loss first"""
def discriminator_minimax_loss(truth, fake, 
							   print_loss = True, 
							   label_smoothing = 0.0,
							   reduction = tf.keras.losses.Reduction.SUM):
	crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits = True, label_smoothing = label_smoothing)
	truth_label = tf.ones_like(truth)
	truth_loss=  crossentropy(truth_label, truth)
	
	fake_label = tf.zeros_like(fake)
	fake_loss= crossentropy(fake_label, fake)
	D_loss = truth_loss + fake_loss  
	if print_loss:
		print("minimax_discriminator_loss: ", tf.reduce_mean(D_loss))
	return D_loss

def cross_entropy_discriminator_loss():
	return True

def wasserstein_discriminator_loss():
	return True

