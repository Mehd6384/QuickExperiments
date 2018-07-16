import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
import gym 

dense = tf.layers.dense


class Policy: 

	def __init__(self): 

		with tf.variable_scope('Perception'): 
			self.state = tf.placeholder(tf.float32, shape = [None,4], name = 'State')

		with tf.variable_scope('Network'): 

			l1 = dense(self.state, 64, activation = tf.nn.relu, name = 'first_layer')
			self.head = dense(l1, 2, activation = tf.nn.softmax, name = 'head')

		with tf.variable_scope('Training'): 

			self.observed_rewards = tf.placeholder(tf.float32, shape = [None, 1], name = 'rewards')
			self.selected_actions = tf.placeholder(tf.int32, shape = [None, 1], name = 'action_indices')


			size = tf.shape(self.observed_rewards)[0]
			indices = tf.reshape(tf.range(size), (-1,1))
			indices = tf.concat([indices, self.selected_actions],1 )
			
		
			self.log_probs = tf.log(tf.gather_nd(self.head, indices))
			self.loss = -self.log_probs*self.observed_rewards

			self.update = tf.train.AdamOptimizer(5e-3).minimize(self.loss)



def discount(r): 

	result, current = [],0.
	for i in reversed(range(len(r))): 
		current = current*0.95 + r[i]
		result.insert(0,current)

	return np.array(result).reshape(-1,1)

import gym 
import time 

env = gym.make('CartPole-v0')

agent = Policy() 
epochs = 700

with tf.Session() as sess: 

	sess.run(tf.global_variables_initializer())


	# grad_buffer = sess.run(tf.trainable_variables())
	# for i,g in enumerate(grad_buffer): 
	# 	grad_buffer[i] *= 0. 

	mean_r = 0. 
	for epoch in range(epochs): 

		s = env.reset()
		reward = 0
		done = False 

		states = []
		rewards = []
		actions = []


		while not done: 

			distrib = sess.run(agent.head, feed_dict = {agent.state:s.reshape(1,-1)}).reshape(-1)
			
			action = np.random.choice(2, p = distrib)

			actions.append(action)
			states.append(s)

			ns, r, done,_ = env.step(action)

			rewards.append(r)
			reward += r
			s = ns 

			if done: 

				mean_r += reward

				discounted = discount(rewards)
				actions = np.array(actions).reshape(-1,1).astype(int)
				states = np.stack(states)


				loss = sess.run(agent.update, feed_dict = {agent.state: states, 
													agent.observed_rewards: discounted,
													agent.selected_actions: actions})

				if epoch%100 == 0: 
					print('Epoch: {} - Reward: {}'.format(epoch, mean_r/100.))
					mean_r = 0.




	for epoch in range(100): 

		s = env.reset()
		done = False 

		while not done: 

			distrib = sess.run(agent.head, feed_dict = {agent.state:s.reshape(1,-1)}).reshape(-1)
			action = np.random.choice(2, p = distrib)

			env.render()
			time.sleep(0.02)

			s, r, done, _ = env.step(action)


