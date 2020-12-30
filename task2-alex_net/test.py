import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import cv2
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from alexnetMnist import Alexnet

# data transform function
# including array to tensor
# and normalization
data_tf = transforms.Compose(
	[# transform to tensor
	 transforms.ToTensor(),
	 # normalization
	 transforms.Normalize([0.5], [0.5])]
)

# prepare dataset
test_set = datasets.MNIST(root='./data',train=False,transform=data_tf)

# load testing dataset
test_loader = DataLoader(test_set,batch_size=1,shuffle=False)

# build alexnet model
# and use gpu to accelerate training
net = Alexnet()
if torch.cuda.is_available():
    net = net.cuda()

# if pre-trained, load the pre-training model
net = torch.load(str(sys.argv[1]))

# evaluation mode
net.eval()

# pretreatment of input image
# load images
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

correct = 0
total = 0

if len(sys.argv) == 2:
	for data in test_loader:
		img,labels = data
		img_arr = img[0][0].numpy()
		img_arr = (img_arr)*255/2
		
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
else:
	if str(sys.argv[2]) == "test/":
		file_list = os.listdir(str(sys.argv[2]))
		for files in file_list:
			print(files)
			img_ori = cv2.imread(str(sys.argv[2])+str(files))
			img_gray = cv2.cvtColor(img_ori,cv2.COLOR_BGR2GRAY)
			img_ori = cv2.resize(img_ori,(28,28),interpolation=cv2.INTER_AREA)
			#img_gray = 255-img_gray
			_,img_gray = cv2.threshold(img_gray,140,255,cv2.THRESH_BINARY_INV)
			#img_gray = cv2.dilate(img_gray, kernel)
			img_gray = cv2.dilate(img_gray, kernel)
			img_gray = cv2.resize(img_gray,(28,28),interpolation=cv2.INTER_AREA)
			#_,img_gray = cv2.threshold(img_gray,150,255,cv2.THRESH_BINARY)
			img_normal = img_gray/255.0

			img_np = np.array(img_normal)
			img_np = np.array(img_np)
			img = data_tf(img_np.astype('float32'))
			img = img.view(1,1,28,28)
			img = img.cuda()
			
			out = net(img)
			_,predicted = torch.max(out.data,1)
			res_np = predicted.cpu().numpy()
			total = total + 1

			img_BGR = cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)

			if str(res_np[0]) == files[0]:
				correct = correct + 1
				img_BGR = cv2.putText(img_BGR,str(res_np[0]),(17,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
			else:
				img_BGR = cv2.putText(img_BGR,str(res_np[0]),(17,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
			print(predicted[0])
			
			
			
			cv2.imwrite('res/'+str(files),img_BGR);
			if total%10 == 1:
				app_col = img_BGR
			else:
				app_col = np.concatenate((app_col,img_BGR),axis=1)
			if total%10 == 0:
				if total == 10:
					app_row = app_col
				else:
					app_row = np.vstack((app_row,app_col))
				cv2.imshow("total",app_row)
				
			cv2.imshow('num',app_col)
			if cv2.waitKey(0) != 32:
				break
	else:
		for i in range(0,10,1):
			file_list = os.listdir(str(sys.argv[2])+str(i)+'/')
			for files in file_list:
				print(files)
				if files[-3:] == 'bmp':
					img_rgb = cv2.imread(str(sys.argv[2])+str(i)+'/'+str(files))
					img_gray = cv2.cvtColor(img_rgb,cv2.COLOR_BGR2GRAY)
					img_gray = 255 - img_gray
					img_normal = img_gray/255.0

					img_np = np.array(img_normal)
					img_np = np.array(img_np)
					img = data_tf(img_np.astype('float32'))
					img = img.view(1,1,28,28)
					img = img.cuda()
					
					out = net(img)
					_,predicted = torch.max(out.data,1)
					total = total + 1
					res_np = predicted.cpu().numpy()
					print(res_np)
					if res_np[0] == i:
						correct = correct + 1

					#cv2.imshow('num',img_gray)
					#if cv2.waitKey(0) != 32:
						#break

print("total img:{}, acc {}%".format(total,correct/total*100))

