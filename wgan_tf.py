import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle

plt.style.use('dark_background')

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


def read_data(path): 

	return pickle.load(open(path, 'rb'))

class Loader: 

	def __init__(self, path, max_el): 

		self.path = path
		self.max_el = max_el

	def sample(self, batch_size): 

		x_ = np.zeros((batch_size, 784))
		# y_ = np.zeros((batch_size, 10))

		inds = np.random.randint(0,self.max_el, (batch_size))

		for i in range(batch_size): 

			x = read_data(self.path + str(inds[i]))
			# label = read_data(self.path + 'labels')[inds[i]]

			x_[i,:] = x
			# y_[i,label] = 1 


		x_ /= 255.
		x_ = (x_ - 0.5)/0.5

		return x_ 


noise_size = 100
batch_size = 64
dense = tf.layers.dense
rms_prop = tf.train.RMSPropOptimizer

noise_g = tf.placeholder(tf.float32, shape = [None, noise_size], name ='real_data')

real = tf.placeholder(tf.float32, shape = [None, 784], name ='real_data')


def generator(fuel, reuse = False): 

	with tf.variable_scope('generator', reuse = reuse): 

		g1 = dense(fuel, 128, activation = tf.nn.relu, name = 'gen_1')
		g2 = dense(g1, 512, activation = tf.nn.relu, name = 'gen_2')
		return dense(g2, 784, activation = None, name = 'gen_output' )

def discriminator(fuel, reuse = False): 

	with tf.variable_scope('discriminator', reuse = reuse): 

		d1 = dense(fuel, 512, activation = tf.nn.relu, name = 'dis_1')
		d2 = dense(d1, 128, activation = tf.nn.relu, name = 'dis_2')
		return dense(d2, 1, activation = None, name = 'dis_out' )


g_samples = generator(noise_g)
d_real = discriminator(real, reuse = False)
d_fake = discriminator(g_samples, reuse = True)


with tf.variable_scope('W_Losses'): 

	d_loss = -(tf.reduce_mean(d_real) - tf.reduce_mean(d_fake)) 
	g_loss = -tf.reduce_mean(d_fake)

all_vars = tf.trainable_variables()
g_vars = [v for v in all_vars if v.name.startswith('generator')]
d_vars = [v for v in all_vars if v.name.startswith('discriminator')]

with tf.variable_scope('training'): 

	train_g = rms_prop(5e-5).minimize(g_loss, var_list = g_vars)
	train_d = rms_prop(5e-5).minimize(d_loss, var_list = d_vars)

loader = Loader('/home/mehdi/Codes/MNIST/', 60000)
f, ax = plt.subplots(2,1)

with tf.Session() as sess: 

	sess.run(tf.global_variables_initializer())

	batch_size = 64
	epochs = 10000
	recap_d = []
	recap_g = []

	for epoch in range(epochs): 

		# train discriminator
		for i in range(np.random.randint(1,5)):
			batch_noise = np.random.normal(0.,1., (batch_size, noise_size))
			batch_x = loader.sample(batch_size)
			loss_d, _ = sess.run([d_loss, train_d], feed_dict = {noise_g:batch_noise, real:batch_x})


		# train generator 
		for i in range(np.random.randint(1,2)): 

			batch_noise = np.random.normal(0.,1., (batch_size, noise_size))
			batch_x = loader.sample(batch_size)

			loss_g,_ = sess.run([g_loss, train_g], feed_dict = {noise_g:batch_noise, real:batch_x})

		recap_g.append(loss_g)
		recap_d.append(loss_d)

		if epoch%50 == 0: 
			print('\n \t\t\t === Epoch {} ===\nLoss D: {:.6f} \nLoss G: {:.6f}\n'.format(epoch, 
				np.mean(recap_d[-20:]), np.mean(recap_g[-20:])))

		if epoch% 25 == 0: 
			for a in ax: 
				a.clear()
			ax[0].plot(recap_g, label = 'Generator')
			ax[0].plot(recap_d, label = 'Discriminator')

			ax[0].legend()

			batch_noise = np.random.normal(0.,1., (batch_size, noise_size))
			batch_x = loader.sample(batch_size)

			
			prod = sess.run(g_samples, feed_dict = {noise_g:batch_noise})

			grid = make_grid(prod.reshape(batch_size,1,28,28))
			ax[1].matshow(grid[:,:,0])


			plt.pause(0.1)

	plt.show()