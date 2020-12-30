import torch
import torch.nn as nn
import torch.nn.functional as F

class Alexnet(nn.Module):
	def __init__(self):
		super(Alexnet,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3,96,11,stride=4,padding=0),
			nn.ReLU(),
			nn.MaxPool2d(3,stride=2)
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(96,256,5,stride=1,padding=2),
			nn.ReLU(),
			nn.MaxPool2d(3,stride=2)
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(256,384,3,stride=1,padding=1),
			nn.ReLU()
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(384,384,3,stride=1,padding=1),
			nn.ReLU()
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(384,256,3,stride=1,padding=1),
			nn.ReLU(),
			nn.MaxPool2d(3,stride=2)
		)
		self.dense = nn.Sequential(
			#nn.Conv2d(256,9216,6,stride=1,padding=0),
			nn.Linear(9216, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10)
		)

	def forward(self,x):
		layer1 = self.conv1(x)
		layer2 = self.conv2(layer1)
		layer3 = self.conv3(layer2)
		layer4 = self.conv4(layer3)
		layer5 = self.conv5(layer4)
		res = layer5.view(layer5.size(0),-1)
		out = self.dense(res)
		return out
