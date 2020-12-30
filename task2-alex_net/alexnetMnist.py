import torch
import torch.nn as nn
import torch.nn.functional as F

class Alexnet(nn.Module):
	def __init__(self):
		super(Alexnet,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(1,32,3,stride=1,padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(32),
			nn.MaxPool2d(2,stride=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(32,64,3,stride=1,padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(64),
			nn.MaxPool2d(2,stride=2)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64,128,3,stride=1,padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU()
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(128,256,3,stride=1,padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU()
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(256,256,3,stride=1,padding=1),
			nn.ReLU(),
			nn.BatchNorm2d(256),
			nn.MaxPool2d(3,stride=2),
			nn.Conv2d(256,256,3,stride=1,padding=0),
			nn.ReLU(),
			nn.BatchNorm2d(256),
		)
		self.dense = nn.Sequential(
			nn.Linear(256, 1024),
			nn.ReLU(),
			nn.BatchNorm1d(1024),
			nn.Dropout(0.5),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.BatchNorm1d(512),
			nn.Dropout(0.5),
			nn.Linear(512, 10)
		)

	def forward(self,x):
		layer1 = self.conv1(x)
		layer2 = self.conv2(layer1)
		layer3 = self.conv3(layer2)
		layer4 = self.conv4(layer3)
		layer5 = self.conv5(layer4)
		res = layer5.view(layer5.size()[0],-1)
		out = self.dense(res)
		return out
