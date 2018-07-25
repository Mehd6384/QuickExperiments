import tensorflow as tf 
import numpy as np 
import pickle 

import matplotlib.pyplot as plt 
plt.style.use('dark_background')

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

code_size = 16 
dense = tf.layers.dense
adam = tf.train.AdamOptimizer

class VAE(object): 

	def __init__(self, sess): 

		self.x = tf.placeholder(name = 'x', dtype = tf.float32, shape = [None, 784])
		with tf.variable_scope('encoder'): 
			e = dense(self.x, 512, tf.nn.relu, name = 'e1')
			e = dense(e, 384, tf.nn.relu, name = 'e2')
			e = dense(e, 256, tf.nn.relu, name = 'e3')

		with tf.variable_scope('code'): 

			self.mu = dense(e, code_size)
			self.logvar = dense(e, code_size)

			eps = tf.random_normal(shape = tf.shape(self.mu))
			self.z = self.mu + tf.sqrt(tf.exp(self.logvar))*eps

		with tf.variable_scope('decoder'): 

			d1 = dense(self.z, 256, tf.nn.relu, name = 'd1')
			d1 = dense(d1, 384, tf.nn.relu, name = 'd2')
			d1 = dense(d1, 512, tf.nn.relu, name = 'd3')
			self.recon = dense(d1, 784, name ='reconstruction', activation = tf.nn.sigmoid)

		with tf.variable_scope('losses'): 

			epsilon = 1e-10
			recon_loss = -tf.reduce_sum(self.x*tf.log(epsilon + self.recon) + (1.-self.x)*tf.log(epsilon + 1-self.recon), axis = 1)
			kld_loss =  -0.5*tf.reduce_sum(1+self.logvar - tf.square(self.mu) - tf.exp(self.logvar), axis = 1)

			self.recon_loss = tf.reduce_mean(recon_loss)
			self.kl_loss = tf.reduce_mean(kld_loss)

			self.total_loss = tf.reduce_mean(recon_loss) + tf.reduce_mean(kld_loss)

			self.train = adam(3e-4).minimize(tf.reduce_mean(recon_loss + kld_loss))

		self.sess = sess

		self.sess.run(tf.global_variables_initializer())

	def run_training_step(self, x): 

		loss,rloss, klloss, _ = self.sess.run([self.total_loss, self.recon_loss, self.kl_loss, self.train], feed_dict = {self.x:x})
		return loss,rloss, klloss 

	def reconstructor(self, x): 
		return self.sess.run(self.recon, feed_dict = {self.x:x})
	
	def generator(self,x): 
		return self.sess.run(self.recon, feed_dict = {self.z:x})
	def transformer(self, x): 
		return self.sess.run(self.z, feed_dict = {self.x:x})


loader = Loader('/home/mehdi/Codes/MNIST/', 60000)

vae = VAE(tf.Session())



def trainer(epochs): 


	for epoch in range(epochs): 

		x = loader.sample(64)
		losses = vae.run_training_step(x)

		if epoch%100 == 0: 
			print('Epoch: {} | Loss : {}'.format(epoch, losses))


trainer(1500)

x = loader.sample(32)
z = np.random.normal(0.,1., (32,code_size))

reconstructed = vae.reconstructor(x)
generated = vae.generator(z)

image = np.concatenate([reconstructed, generated], 0)
image = image.reshape(64,1,28,28)


image = make_grid(image)

plt.imshow(image)
plt.show()
