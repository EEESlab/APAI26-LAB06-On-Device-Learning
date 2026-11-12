import numpy as np
import dump_utils as dump
import math

# Define hyperparameters
learning_rate = 0.1			# CHANGE ME
batch_size = 1				# DO NOT CHANGE ME
epochs = 1					# CHANGE ME

# LAYER 0 SIZES
l0_in_ch = 3
l0_out_ch = 8
l0_hk = 3
l0_wk = 3
l0_hin = 8
l0_win = 8
l0_hstr = 1
l0_wstr = 1
l0_hpad = 0
l0_wpad = 0

# LAYER 1 SIZES
l1_in_ch = 8
l1_out_ch = 8
l1_hk = 1
l1_wk = 1
l1_hin = 6
l1_win = 6
l1_hstr = 1
l1_wstr = 1
l1_hpad = 0
l1_wpad = 0

# LAYER 2 SIZES
l2_in_ch = 288
l2_out_ch = 10
l2_hk = 1
l2_wk = 1
l2_hin = 1
l2_win = 1
l2_hstr = 1
l2_wstr = 1
l2_hpad = 0
l2_wpad = 0

f = open('init-defines.h', 'w')
f.write('// Layer0\n')
f.write('#define Tin_C_l0 '+str(l0_in_ch)+'\n')
f.write('#define Tout_C_l0 '+str(l0_out_ch)+'\n')
f.write('#define Tker_H_l0 '+str(l0_hk)+'\n')
f.write('#define Tker_W_l0 '+str(l0_wk)+'\n')
f.write('#define Tin_H_l0 '+str(l0_hin)+'\n')
f.write('#define Tin_W_l0 '+str(l0_win)+'\n')
f.write('#define Tout_H_l0 '+str(math.floor((l0_hin-l0_hk+2*l0_hpad+l0_hstr)/l0_hstr))+'\n')
f.write('#define Tout_W_l0 '+str(math.floor((l0_win-l0_wk+2*l0_wpad+l0_wstr)/l0_wstr))+'\n')
f.write('#define Tstr_H_l0 '+str(l0_hstr)+'\n')
f.write('#define Tstr_W_l0 '+str(l0_wstr)+'\n')
f.write('#define Tpad_H_l0 '+str(l0_hpad)+'\n')
f.write('#define Tpad_W_l0 '+str(l0_wpad)+'\n')

f.write('// Layer1\n')
f.write('#define Tin_C_l1 '+str(l1_in_ch)+'\n')
f.write('#define Tout_C_l1 '+str(l1_out_ch)+'\n')
f.write('#define Tin_H_l1 '+str(l1_hin)+'\n')
f.write('#define Tin_W_l1 '+str(l1_win)+'\n')
f.write('#define Tout_H_l1 '+str(math.floor((l1_hin-l1_hk+2*l1_hpad+l1_hstr)/l1_hstr))+'\n')
f.write('#define Tout_W_l1 '+str(math.floor((l1_win-l1_wk+2*l1_wpad+l1_wstr)/l1_wstr))+'\n')
f.write('#define Tstr_H_l1 '+str(l1_hstr)+'\n')
f.write('#define Tstr_W_l1 '+str(l1_wstr)+'\n')
f.write('#define Tpad_H_l1 '+str(l1_hpad)+'\n')
f.write('#define Tpad_W_l1 '+str(l1_wpad)+'\n')

f.write('// Layer2\n')
f.write('#define Tin_C_l2 '+str(l2_in_ch)+'\n')
f.write('#define Tout_C_l2 '+str(l2_out_ch)+'\n')
f.write('#define Tker_H_l2 '+str(l2_hk)+'\n')
f.write('#define Tker_W_l2 '+str(l2_wk)+'\n')
f.write('#define Tin_H_l2 '+str(l2_hin)+'\n')
f.write('#define Tin_W_l2 '+str(l2_win)+'\n')
f.write('#define Tout_H_l2 '+str(math.floor((l2_hin-l2_hk+2*l2_hpad+l2_hstr)/l2_hstr))+'\n')
f.write('#define Tout_W_l2 '+str(math.floor((l2_win-l2_wk+2*l2_wpad+l2_wstr)/l2_wstr))+'\n')
f.write('#define Tstr_H_l2 '+str(l2_hstr)+'\n')
f.write('#define Tstr_W_l2 '+str(l2_wstr)+'\n')
f.write('#define Tpad_H_l2 '+str(l2_hpad)+'\n')
f.write('#define Tpad_W_l2 '+str(l2_wpad)+'\n')
f.close()

f = open('init-defines.h', 'a')
f.write('\n// HYPERPARAMETERS\n')
f.write('#define LEARNING_RATE '+str(learning_rate)+'\n')
f.write('#define EPOCHS '+str(epochs)+'\n')
f.write('#define BATCH_SIZE '+str(batch_size)+'\n')
f.close()

# Simple input data
inp = np.ones((batch_size, l0_in_ch, l0_hin, l0_win), dtype=np.float32) / np.float32(1000)


# ---------------------------------------------------------------------------
# Layer primitives (NumPy re-implementation of the former PyTorch modules)
# ---------------------------------------------------------------------------

def conv2d_forward(x, w, stride, pad):
	"""2D convolution without bias. x: (B, Cin, H, W), w: (Cout, Cin, Kh, Kw)."""
	sh, sw = stride
	ph, pw = pad
	xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
	# windows: (B, Cin, Hout', Wout', Kh, Kw) -> apply stride
	win = np.lib.stride_tricks.sliding_window_view(xp, w.shape[2:], axis=(2, 3))[:, :, ::sh, ::sw]
	return np.einsum('bchwij,ocij->bohw', win, w).astype(np.float32)


def conv2d_weight_grad(x, grad_out, w_shape, stride, pad):
	"""Gradient of the conv2d loss w.r.t. its weights."""
	sh, sw = stride
	ph, pw = pad
	xp = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)))
	win = np.lib.stride_tricks.sliding_window_view(xp, w_shape[2:], axis=(2, 3))[:, :, ::sh, ::sw]
	return np.einsum('bchwij,bohw->ocij', win, grad_out).astype(np.float32)


class DNN:
	def __init__(self):
		# Weights initialized as N(0, 1), as with nn.init.normal_(mean=0, std=1)
		self.l0_weight = np.random.normal(0.0, 1.0, (l0_out_ch, l0_in_ch, l0_hk, l0_wk)).astype(np.float32)
		self.l2_weight = np.random.normal(0.0, 1.0, (l2_out_ch, l2_in_ch)).astype(np.float32)

	def forward(self, x):
		self.x0 = x
		self.z0 = conv2d_forward(x, self.l0_weight, (l0_hstr, l0_wstr), (l0_hpad, l0_wpad))	# l0: Conv2D
		self.a1 = np.maximum(self.z0, 0)													# l1: ReLU
		self.v = self.a1.reshape(-1)														# flatten
		return (self.l2_weight @ self.v).astype(np.float32)								# l2: Linear

	def backward(self, grad_out):
		# l2: Linear
		self.l2_grad = np.outer(grad_out, self.v).astype(np.float32)
		grad_v = self.l2_weight.T @ grad_out
		# reshape + l1: ReLU
		grad_z0 = grad_v.reshape(self.z0.shape) * (self.z0 > 0)
		# l0: Conv2D
		self.l0_grad = conv2d_weight_grad(self.x0, grad_z0, self.l0_weight.shape, (l0_hstr, l0_wstr), (l0_hpad, l0_wpad))

	def sgd_step(self, lr):
		self.l0_weight -= np.float32(lr) * self.l0_grad
		self.l2_weight -= np.float32(lr) * self.l2_grad


def mse_loss(out, target):
	"""MSE loss (mean reduction) and its gradient w.r.t. out."""
	diff = out - target
	loss = np.mean(diff ** 2)
	grad = (2.0 / diff.size) * diff
	return loss, grad.astype(np.float32)


# Initialize network
net = DNN()

# All-ones fake label 
output_test = net.forward(inp)
label = np.ones_like(output_test)

f = open('io_data.h', 'w')
f.write('// Init weights\n')
f.write('#define WGT_SIZE_L0 '+str(l0_in_ch*l0_out_ch*l0_hk*l0_wk)+'\n')
f.write('PI_L2 float init_WGT_l0[WGT_SIZE_L0] = {'+dump.tensor_to_string(net.l0_weight)+'};\n')
f.write('#define WGT_SIZE_L2 '+str(l2_in_ch*l2_out_ch*l2_hk*l2_wk)+'\n')
f.write('PI_L2 float init_WGT_l2[WGT_SIZE_L2] = {'+dump.tensor_to_string(net.l2_weight)+'};\n')
f.close()

# Train the DNN (SGD, no momentum)
for batch in range(epochs):
	out = net.forward(inp)
	loss, grad = mse_loss(out, label)
	net.backward(grad)
	net.sgd_step(learning_rate)

# Inference once after training
out = net.forward(inp)

f = open('io_data.h', 'a')
f.write('// Input and Output data\n')
f.write('#define IN_SIZE 192\n')
f.write('PI_L1 float INPUT[IN_SIZE] = {'+dump.tensor_to_string(inp)+'};\n')
out_size = (int(math.floor(l2_hin-l2_hk+2*l2_hpad+l2_hstr)/l2_hstr)) * (int(math.floor(l2_win-l2_wk+2*l2_wpad+l2_wstr)/l2_wstr)) * l2_out_ch
f.write('#define OUT_SIZE '+str(out_size)+'\n')
f.write('PI_L2 float REFERENCE_OUTPUT[OUT_SIZE] = {'+dump.tensor_to_string(out)+'};\n')
f.write('PI_L1 float LABEL[OUT_SIZE] = {'+dump.tensor_to_string(label)+'};\n')
f.close()
