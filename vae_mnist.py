import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 

from torchvision.utils import save_image 
from torch.distributions import Normal 

import pickle
import numpy as np 
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
		if use_cuda: 
			tensor = torch.cuda.FloatTensor(data).float()
		else: 
			tensor = torch.from_numpy(data).float()

		return tensor


code_size = 20

class VAE(nn.Module): 


	def __init__(self): 

		nn.Module.__init__(self)

		self.encoder = nn.ModuleList([nn.Linear(784,400), nn.Linear(400,128)])

		self.sigma = nn.Linear(128,code_size)
		self.mu = nn.Linear(128, code_size)

		self.decoder = nn.ModuleList([nn.Linear(code_size, 128), nn.Linear(128,400), nn.Linear(400,784)])

	def forward(self, x): 

		for l in self.encoder: 
			x = F.leaky_relu(l(x), 0.1)
		
		stds = torch.exp(self.sigma(x))
		means = self.mu(x)

		z = self.reparam(means, stds)

		for i, l in enumerate(self.decoder): 
			if i == 0: 
				recon = F.leaky_relu(l(z), 0.1)
			else: 
				recon = F.leaky_relu(l(recon), 0.1)

		return recon, means, stds, z 

	def reparam(self, m, s): 

		if self.training: 
			eps = torch.randn_like(s)
			z = m + s*eps 
		else: 
			z = m 
		return z 

	def sample(self, nb): 

		x = torch.randn(nb, code_size)
		for i, l in enumerate(self.decoder): 
			if i == 0: 
				recon = F.leaky_relu(l(x), 0.1)
			else: 
				recon = F.leaky_relu(l(recon), 0.1)

		return recon 

def loss_fn(x, recon, mean, logvar): 

	recon_loss = F.mse_loss(recon, x, size_average = False)
	kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

	return kl_loss + recon_loss
def normal_init(model, mean = 0., s = 0.02): 

	for m in model._modules: 

		if isinstance(model._modules[m], nn.Linear): 
			model._modules[m].weight.data.normal_(mean ,s)
			model._modules[m].bias.data.zero_()
		elif isinstance(model._modules[m], nn.ModuleList): 

			size = len(model._modules[m])
			for i in range(size): 
				model._modules[m][i].weight.data.normal_(mean ,s)
				model._modules[m][i].bias.data.zero_()


l = Loader('/home/mehdi/Codes/MNIST/', 60000)
x = l.sample()
vae = VAE()

normal_init(vae)

adam = optim.Adam(vae.parameters(), 2e-4)
epochs = 5000
for epoch in range(1,epochs+1): 

	x = l.sample(64)

	recon, m, std, _ = vae(x)

	loss = loss_fn(x, recon, m, std)
	adam.zero_grad()
	loss.backward()
	adam.step()

	if epoch%100 == 0: 

		x = l.sample(32)
		recon,_,_,_ = vae(x)

		samples = vae.sample(32)

		recon = recon.reshape(32, 1, 28,28)
		samples = samples.reshape(32, 1, 28,28)

		image = torch.cat([recon, samples], 0)
		save_image(image, 'results/{}.png'.format(epoch))



x = l.sample(64)
recon,_,_,_ = vae(x)

samples = vae.sample(64)

save_image(recon.reshape(64,1,28,28), 'results/FinalRecon.png'.format(epoch))
save_image(samples.reshape(64,1,28,28), 'results/FinalSamples.png'.format(epoch))