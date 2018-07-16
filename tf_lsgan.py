import tensorflow as tf 
import numpy as np 
import pickle 

import matplotlib.pyplot as plt 
plt.style.use('ggplot')

dense = tf.layers.dense

class Loader: 

	def __init__(self, path, max_points): 

		self.path = path 
		self.max_points = max_points

	def sample(self, batch): 

		data = np.zeros((batch,784))
		index = np.random.randint(0,self.max_points,(batch))
		for i,ind in enumerate(index): 
			im = pickle.load(open(self.path+'{}'.format(ind), 'rb')) 
			data[i,:] = im.reshape(-1)

		data /= 255.
		data = (data - 0.5)/0.5
		return data

noise_size = 50
loader = Loader('/home/mehdi/Codes/MNIST/', 60000)

def Generator(input_var, reuse = False):

	with tf.variable_scope('Generator', reuse = reuse): 

		# input_var = tf.placeholder(tf.float32, shape = [None, noise_size], name = 'latent_variable')
		y_gen = tf.placeholder(tf.float32, shape = [None, 1], name = 'truth_measure')

		l1 = dense(input_var, 128, activation = tf.nn.leaky_relu, name = 'first_generator_layer', kernel_initializer = tf.variance_scaling_initializer)
		l2 = dense(l1, 400, activation = tf.nn.leaky_relu, name = 'second_generator_layer', kernel_initializer = tf.variance_scaling_initializer)
		generator = dense(l2, 784, activation = None, name = 'generator')

		return generator

def Discriminator(input_var, reuse= False): 
	with tf.variable_scope('Discriminator', reuse = reuse): 

		# x = tf.placeholder(tf.float32, shape = [None, 784], name = 'obs_variable')
		y_dis = tf.placeholder(tf.float32, shape = [None, 1], name = 'labels')

		l1 = dense(input_var, 400, activation = tf.nn.leaky_relu, name = 'first_generator_layer', kernel_initializer = tf.variance_scaling_initializer)
		l2 = dense(l1, 128, activation = tf.nn.leaky_relu, name = 'second_generator_layer', kernel_initializer = tf.variance_scaling_initializer)
		discriminator = dense(l2, 1, activation = None, name = 'generator')

		return discriminator


z = tf.placeholder(tf.float32, shape = [None, noise_size], name = 'latent_variable')
x = tf.placeholder(tf.float32, shape = [None, 784], name = 'real_data')

g = Generator(z)
d_real = Discriminator(x)
d_fake = Discriminator(g, reuse = True)

d_loss = tf.reduce_mean(tf.squared_difference(d_real, tf.ones_like(d_real)) + tf.pow(d_fake,2))
g_loss = tf.reduce_mean(tf.squared_difference(d_fake, tf.ones_like(d_fake)))

variables = tf.trainable_variables()
g_vars = [v for v in variables if 'Generator' in v.name] 
d_vars = [v for v in variables if 'Discriminator' in v.name] 


train_g = tf.train.AdamOptimizer(2e-4).minimize(g_loss, var_list = g_vars)
train_d = tf.train.AdamOptimizer(1e-4).minimize(d_loss, var_list = d_vars)


epochs = 10000
batch_size = 64 
with tf.Session() as sess: 

	sess.run(tf.global_variables_initializer())
	mean_loss_d = 0.
	mean_loss_g = 0.
	for epoch in range(1,epochs+1): 

		for i in range(np.random.randint(1,3)):
			d_x = loader.sample(batch_size)
			noise = np.random.normal(0,1.,(batch_size, noise_size))

			# train D 

			_ , d_loss_var = sess.run([train_d, d_loss], feed_dict = {x:d_x, z:noise})
			mean_loss_d += d_loss_var

		# train G 
		for i in range(np.random.randint(1,4)): 

			noise = np.random.normal(0,1.,(batch_size, noise_size))
			_ , g_loss_var = sess.run([train_g, g_loss], feed_dict = {z:noise})

		mean_loss_g += g_loss_var
		
		if epoch% 100 == 0: 
			print('\t\t\t === Epoch: {} === \n\nLoss D: {:.6f}\nLoss G: {:.6f}'.format(epoch, mean_loss_d/100., mean_loss_g/100.))
			mean_loss_d = 0.
			mean_loss_g = 0.

	over = False 
	f, ax = plt.subplots()
	while not over: 

		noise = np.random.normal(0,1.,(1, noise_size))
		prod = sess.run(g, feed_dict = {z:noise})
		ax.clear()

		ax.matshow(prod.reshape(28,28))
		plt.pause(0.1)
		over = input('Press o to quit')

		over = True if over == 'o' else False 

