import numpy as np 
import tensorflow as tf 
import pickle 
import matplotlib.pyplot as plt 

from PIL import Image 


class Loader():

	def __init__(self, path = '', max_items = 60000): 

		self.path = path
		self.max_count = max_items
		self.counter = 0 

	def sample(self,batch_size = 32, use_cuda = False): 

		data = np.zeros((batch_size, 784))
		ind = np.random.randint(0,self.max_count, (batch_size))
		for i in range(batch_size): 
			name = self.path+'{}'.format(ind[i])
			image = pickle.load(open(name,'rb'))
			data[i,:] = image.reshape(784)
			self.counter = (self.counter+1)%self.max_count

		data /= 255. 

		return data

def make_grid(x, rows = 4, decal = 2): 

	# x is a (batch_size, nb_channels, height, width) nd-array

	col = int(x.shape[0]/rows)
	image = np.zeros(((x.shape[2] + decal)*rows, (decal + x.shape[3])*col, 3))
	ligne = 0 
	column = 0 
	for i in range(x.shape[0]): 
		current = x[i,:,:,:]
		current = np.transpose(current, [1,2,0])

		
		image[decal + ligne*(x.shape[2]):(ligne+1)*(x.shape[2]) + decal, decal + column*(x.shape[3]):(column+1)*(x.shape[3]) + decal,:] = current

		column = (column + 1)%col
		if(column == 0): 
			ligne += 1

	# input(image.shape)
	return image 

dense = tf.layers.dense 
# relu = tf.nn.relu(alpha = 0.2)

code_size = 20 
x = tf.placeholder(tf.float32, shape = [None, 784], name = 'Input_x')

with tf.variable_scope('Encoder'): 
	e1 = dense(x, 512, activation = tf.nn.relu, name = 'encoding_1')
	e1 = dense(e1, 384, activation = tf.nn.relu, name = 'encoding_2')
	e1= dense(e1, 256, activation = tf.nn.relu, name = 'encoding_3')

with tf.variable_scope('Reparametrization'): 
	z_means = dense(e1, code_size, name = 'means')
	z_stds = dense(e1, code_size, name = 'logvar')

with tf.variable_scope('Code'): 
	z = z_means + tf.random_normal(tf.shape(z_stds))*tf.exp(z_stds)


with tf.variable_scope('Decoder'): 
	d1 = dense(z, 256, activation = tf.nn.relu, name = 'decoding_1')
	d1 = dense(d1, 384, activation = tf.nn.relu, name = 'decoding_2')
	d2 = dense(d1, 512, activation = tf.nn.relu, name = 'decoding_3')
	out = dense(d2, 784, name = 'reconstruction', activation = tf.nn.sigmoid)


with tf.variable_scope('Losses'): 

	kl_loss = -0.5*tf.reduce_sum(1. + z_stds - tf.pow(z_means,2) - tf.exp(z_stds), axis = 1)
	# recon_loss = tf.losses.mean_squared_error(x, out, reduction_indices = 1)	
	recon_loss = tf.reduce_sum(tf.square(out - x), axis = 1)

with tf.variable_scope('Training'): 

	full_loss = tf.reduce_mean(kl_loss + recon_loss)
	update_vae = tf.train.AdamOptimizer(5e-4).minimize(full_loss)



epochs = 2500
loader = Loader('/home/mehdi/Codes/MNIST/',60000)

f, ax = plt.subplots(1,2)

with tf.Session() as sess: 

	sess.run(tf.global_variables_initializer())

	mean_loss = 0. 
	for epoch in range(1,epochs+1): 

		data_x = loader.sample(64)

		loss, _ = sess.run([full_loss, update_vae], feed_dict = {x:data_x})
		mean_loss += np.mean(loss)

		if epoch% 100 == 0: 
			# input(loss)
			print('Epoch: {} | Loss: {:.6f}'.format(epoch, mean_loss/100.))
			mean_loss = 0.

		if epoch % 100 == 0: 

			data_x = loader.sample(32)
			result = sess.run([out], feed_dict = {z:np.random.normal(0.,1., (32,code_size))})[0]
			recons = sess.run([out], feed_dict = {x:data_x})[0]

			for a in ax: 
				a.clear()

			ax[0].imshow(make_grid(recons.reshape(32,1,28,28)))
			ax[1].imshow(make_grid(result.reshape(32,1,28,28)))

			ax[0].set_title('Recons')
			ax[1].set_title('Prods')

			plt.pause(0.1)
	
	plt.show()
