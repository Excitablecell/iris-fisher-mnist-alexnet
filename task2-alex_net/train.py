import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from alexnetMnist import Alexnet
from transform import elastic_tf

# training parameters
batch_size = 128
learning_rate = 1e-3
num_epoches = 40

# data transform function
# including array to tensor
# and normalization
train_tf = transforms.Compose(
	[# random flip
	 transforms.RandomRotation(30),
	 transforms.RandomResizedCrop(28),
	 # transform to tensor
	 transforms.ToTensor(),
	 # normalization
	 transforms.Normalize([0.5], [0.5])
	])

test_tf = transforms.Compose(
	[# random flip
	 # transforms.RandomHorizontalFlip(),
	 # transforms.RandomResizedCrop(28),
	 # transform to tensor
	 transforms.ToTensor(),
	 # normalization
	 transforms.Normalize([0.5], [0.5])
	])

# prepare dataset
train_db_t = datasets.MNIST(root='./data',train=True,transform=train_tf)
train_db_o = datasets.MNIST(root='./data',train=True,transform=test_tf)
train_db_e = datasets.MNIST(root='./data',train=True,transform=test_tf)
train_db = train_db_t + train_db_o
test_set = datasets.MNIST(root='./data',train=False,transform=test_tf)
cnt = 0
for train_iter in train_db_e:
	img,_ = train_iter
	train_db_e[cnt][0][0] = elastic_tf(img[0])
	cnt = cnt + 1

# load test set
test_loader = DataLoader(test_set,batch_size=batch_size,shuffle=False)

# build alexnet model
# and use gpu to accelerate training
net = Alexnet()
if torch.cuda.is_available():
    net = net.cuda()

# use cross entropy as loss function
# and use SGD as optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=learning_rate)

# if pre-trained, load the pre-training model
# net = torch.load('MNIST_Crop_adam.pk1')

# below is the training process
epoch = 0
loss_show = []
acc_show = []
acc_axis = []
count = 0

correct = 0
total = 0

for epoch in range(num_epoches):
	# random split dataset into train set and valuating set
	train_set, val_set = torch.utils.data.random_split(train_db, [50000*2, 10000*2])

	# load training dataset
	train_loader = DataLoader(train_set+train_db_e,batch_size=batch_size,shuffle=True)
	val_loader = DataLoader(val_set,batch_size=batch_size,shuffle=True)
	
	for data in train_loader:
		img,label = data
		
		# randomly elastic transform
		rand = np.random.randint(0,32-1,10)
		for i in range(0,10,1):
			img[rand[i]] = elastic_tf(img[rand[i]])

		if torch.cuda.is_available():
			img = img.cuda()
			labels = label.cuda()
		else:
			print('cuda is unavailable!')
			img = Variable(img)
			labels = Variable(label)

		# forward propagation
		out = net(img)
		loss = criterion(out,labels)
		loss_show.append(loss)
		count = count + 1

		# back propagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	for data in val_loader:
		img,label = data
		if torch.cuda.is_available():
			img = img.cuda()
			labels = label.cuda()
		else:
			print('cuda is unavailable!')
			img = Variable(img)
			labels = Variable(label)

		# forward propagation
		out = net(img)
		_,predicted = torch.max(out.data,1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

	# finish a epoch of training
	epoch += 1
	print('epoch: {}, loss: {:.4}'.format(epoch, loss.data.item()))
	print('train accuracy:{}%'.format(100*correct/total))
	acc_show.append(correct/total)
	acc_axis.append(count)
	total = 0
	correct = 0
	# save the model
	torch.save(net,'MNIST_Crop_adam_3.pk1')

print('Training done!')
plt.plot(list(range(1,len(loss_show)+1,1)),loss_show,label='train loss')
plt.plot(acc_axis,acc_show,label='train accuracy')


correct = 0
total = 0

for data in test_loader:
	img,labels = data
	if torch.cuda.is_available():
			img = img.cuda()
			labels = labels.cuda()
	else:
		print('cuda is unavailable!')
		img = Variable(img)
		labels = Variable(labels)

	# forward propagation
	out = net(img)
	_,predicted = torch.max(out.data,1)
	total += labels.size(0)
	correct += (predicted == labels).sum().item()

print('test accuracy:{}%'.format(100*correct/total))
plt.legend()
plt.show()

