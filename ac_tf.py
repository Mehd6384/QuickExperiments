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

			self.advantage = tf.placeholder(tf.float32, shape = [None, 1], name = 'rewards')
			self.selected_actions = tf.placeholder(tf.int32, shape = [None, 1], name = 'action_indices')


			size = tf.shape(self.advantage)[0]
			indices = tf.reshape(tf.range(size), (-1,1))
			indices = tf.concat([indices, self.selected_actions],1 )
			
		
			self.log_probs = tf.log(tf.gather_nd(self.head, indices))
			self.loss = -self.log_probs*self.advantage

			self.update = tf.train.AdamOptimizer(5e-3).minimize(self.loss)

class Critic: 

	def __init__(self):

		with tf.variable_scope('Critic_Perception'): 
			self.state = tf.placeholder(tf.float32, shape = [None,4], name = 'Critic_State')

		with tf.variable_scope('Critic_Network'): 

			l1 = dense(self.state, 64, activation = tf.nn.relu, name = 'Critic_first_layer')
			self.head = dense(l1, 1, activation = None, name = 'Critic_head')

		with tf.variable_scope('Critic_Training'): 

			self.targets = tf.placeholder(tf.float32, shape = [None, 1], name = 'targets')

			self.loss = tf.squared_difference(self.head, self.targets)

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
critic = Critic()

epochs = 700

with tf.Session() as sess: 

	sess.run(tf.global_variables_initializer())

	mean_r = 0. 
	for epoch in range(epochs): 

		s = env.reset()
		reward = 0
		done = False 

		states = []
		rewards = []
		actions = []
		values = []

		while not done: 

			distrib, estim = sess.run([agent.head, critic.head], feed_dict = {agent.state:s.reshape(1,-1), critic.state:s.reshape(1,-1)})
			action = np.random.choice(2, p = distrib.reshape(-1))

			values.append(estim.reshape(1))
			actions.append(action)
			states.append(s)

			ns, r, done,_ = env.step(action)

			rewards.append(r)
			reward += r
			s = ns 

			if done: 

				mean_r += reward

				discounted = discount(rewards).reshape(-1,1)
				actions = np.array(actions).reshape(-1,1).astype(int)
				states = np.stack(states)
				values = np.stack(values).reshape(-1,1)

				# Update critic

				critic_loss,_ = sess.run([critic.loss, critic.update], feed_dict = {critic.state: states,
																   critic.targets: discounted})
				print(np.mean(critic_loss.reshape(-1)))

				loss = sess.run(agent.update, feed_dict = {agent.state: states, 
													agent.advantage: discounted - values,
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


