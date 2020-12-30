import numpy as np
import cv2
import torch
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
	if random_state is None:
		random_state = np.random.RandomState(None)

	shape = image.shape
	shape_size = shape[:2]

    # Random affine
	center_square = np.float32(shape_size) // 2
	square_size = min(shape_size) // 3
	pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
	pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
	M = cv2.getAffineTransform(pts1, pts2)
	image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

	dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
	dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
	dz = np.zeros_like(dx)

	x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
	indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

	return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
	#return image

def elastic_tf(tensor):
	img = tensor.numpy()[0]
	img = img * 255
	img_bgr = cv2.merge([img,img,img])
	img2 = elastic_transform(img_bgr,34,4,1)
	img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	tensor = torch.from_numpy(img2/255)
	return tensor

if __name__ == '__main__':
	load_data = datasets.MNIST(root='./data',train=True,transform=transforms.ToTensor())
	datas = DataLoader(load_data,batch_size=2,shuffle=True)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	count = 0
	for data in datas:
		tensor,label = data
		img = tensor[0].numpy()[0]
		img = img * 255
		img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
		img2 = elastic_transform(img,34,4,1)
		img = cv2.putText(img,'1',(17,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,255,0),1)
		img2 = cv2.putText(img2,'2',(17,10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1)
		#img2 = cv2.erode(img2, kernel)
		img_comp = np.concatenate((img,img2),axis=1)
		count = count + 1
		if count %5 == 1:
			app_col = img_comp
		else:
			app_col = np.concatenate((app_col,img_comp),axis=1)
		if count%5 == 0:
			if count == 5:
				app_row = app_col
			else:
				app_row = np.vstack((app_row,app_col))
			cv2.imshow('row',app_row)
		cv2.imshow('col',app_col)
		if cv2.waitKey(0) != 32:
			break

