'''
Copyright (C) 2021-2022 ETH Zurich and University of Bologna

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import numpy as np

def tensor_to_string(tensor):
	"""Flatten an N-D array (row-major / C order) into a C initializer string."""
	arr = np.asarray(tensor, dtype=np.float32)
	print("NDIM", arr.ndim)
	if arr.ndim > 1:
		print('Sizes: ', *arr.shape)

	tensor_string = ''
	for v in arr.reshape(-1):
		tensor_string += str(float(v))
		tensor_string += 'f, '
	return tensor_string



def main():
	import argparse
	parser = argparse.ArgumentParser("FCN Layer Test")
	parser.add_argument( '--in_size', type=int, default=2,
	    help="An integer will be increased by 1 and printed." )
	parser.add_argument( '--out_size', type=int, default=2,
	    help="An integer will be increased by 1 and printed." )
	args = parser.parse_args()

	dim0_sz = args.in_size
	dim1_sz = args.out_size
	t = np.random.rand(dim0_sz).astype(np.float32)
	print(t)
	print(tensor_to_string(t))

	t = np.random.rand(dim1_sz, dim0_sz).astype(np.float32)
	print(t)
	print(tensor_to_string(t))


if __name__ == '__main__':
    main()
