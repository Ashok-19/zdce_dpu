import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.jit import script

'''
@script
def _ts_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * y'''

class CSDN_Tem(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(CSDN_Tem, self).__init__()
		self.depth_conv = nn.Conv2d(
			in_channels=in_ch,
			out_channels=in_ch,
			kernel_size=3,
			stride=1,
			padding=1,
			groups=in_ch
		)
		self.point_conv = nn.Conv2d(
			in_channels=in_ch,
			out_channels=out_ch,
			kernel_size=1,
			stride=1,
			padding=0,
			groups=1
		)

	def forward(self, input):
		out = self.depth_conv(input)
		out = self.point_conv(out)
		return out

class enhance_net_nopool(nn.Module):

	def __init__(self,scale_factor):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=True)
		self.scale_factor = scale_factor
		self.upsample = nn.Upsample(scale_factor=4,mode='bilinear')
		self.upsample_down = nn.Upsample(scale_factor=0.25, mode='bilinear')
		number_f = 32

#   zerodce DWC + p-shared
		self.e_conv1 = CSDN_Tem(3,number_f) 
		self.e_conv2 = CSDN_Tem(number_f,number_f) 
		self.e_conv3 = CSDN_Tem(number_f,number_f) 
		self.e_conv4 = CSDN_Tem(number_f,number_f) 
		self.e_conv5 = CSDN_Tem(number_f*2,number_f) 
		self.e_conv6 = CSDN_Tem(number_f*2,number_f) 
		self.e_conv7 = CSDN_Tem(number_f*2,3) 

	def enhance(self, x,x_r):

		#mul = _ts_mul #if torch.jit.is_scripting() else custom_mul
		
		x1 = x
		x2 = x + 0.00000001
		#1st
		x_sq = torch.mul(x1,x2)
		diff = x_sq + (-1)*x
		alpha = torch.mul(x_r, diff)
		enhanced = x + alpha
		x = x + x_r*(torch.pow(x,2)-x)
		
		#2nd
		x1 = enhanced
		x2 = enhanced + 0.00000001
		x_sq = torch.mul(x1,x2)
		diff = x_sq + (-1)*x
		alpha = torch.mul(x_r, diff)
		enhanced = x + alpha
		
		#3rd
		x1 = enhanced
		x2 = enhanced + 0.00001
		x_sq = torch.mul(x1,x2)
		diff = x_sq + (-1)*x
		alpha = torch.mul(x_r, diff)
		enhanced = x + alpha
		
		#4th
		x1 = enhanced
		x2 = enhanced + 0.00000001
		x_sq = torch.mul(x1,x2)
		diff = x_sq + (-1)*x
		alpha = torch.mul(x_r, diff)
		enhanced = x + alpha
		
		#5th
		x1 = enhanced
		x2 = enhanced + 0.00000001
		x_sq = torch.mul(x1,x2)
		diff = x_sq + (-1)*x
		alpha = torch.mul(x_r, diff)
		enhanced = x + alpha
		
		#6th
		x1 = enhanced
		x2 = enhanced + 0.00000001
		x_sq = torch.mul(x1,x2)
		diff = x_sq + (-1)*x
		alpha = torch.mul(x_r, diff)
		enhanced = x + alpha
		
		#7th
		x1 = enhanced
		x2 = enhanced + 0.00000001
		x_sq = torch.mul(x1,x2)
		diff = x_sq + (-1)*x
		alpha = torch.mul(x_r, diff)
		enhanced = x + alpha
		
		#8th
		x1 = enhanced
		x2 = enhanced + 0.00000001
		x_sq = torch.mul(x1,x2)
		diff = x_sq + (-1)*x
		alpha = torch.mul(x_r, diff)
		enhanced = x + alpha
			
		return enhanced

		'''
		x = x + x_r*(torch.pow(x,2)-x)
		x = x + x_r*(torch.pow(x,2)-x)
		x = x + x_r*(torch.pow(x,2)-x)
		enhance_image_1 = x + x_r*(torch.pow(x,2)-x)		
		x = enhance_image_1 + x_r*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + x_r*(torch.pow(x,2)-x)	
		x = x + x_r*(torch.pow(x,2)-x)
		enhance_image = x + x_r*(torch.pow(x,2)-x)	'''

		return enhance_image
		
	def forward(self, x):
		if self.scale_factor==1:
			x_down = x
		else:
			x_down = self.upsample_down(x)

		x1 = self.relu(self.e_conv1(x_down))
		x2 = self.relu(self.e_conv2(x1))
		x3 = self.relu(self.e_conv3(x2))
		x4 = self.relu(self.e_conv4(x3))
		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		x_r = F.hardsigmoid(self.e_conv7(torch.cat([x1,x6], dim=1)), inplace=False) 
		x_r = 2*x_r + (-1)
		
		if self.scale_factor==1:
			x_r = x_r
		else:
			x_r = self.upsample(x_r)
		enhance_image = self.enhance(x,x_r)
		return enhance_image,x_r
