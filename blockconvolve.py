import numpy as np
import matplotlib.pyplot as plt


# load image
array = np.load('array.npy')

rows, cols = array.shape

x = 4
y = 2

x_dim = int(rows/x)
y_dim = int(cols/y)

kernel = np.full((x,y), 1/(x*y))

output = np.zeros((x_dim, y_dim))

# block convultion; image is 32x32, kernel is 4x4 and moves over 4 px every time to ensure no overlap

for i in range(0, rows, x):
    for j in range(0, cols, y):
        region = array[i:i+x, j:j+y] # slices a 4x4 section of the the array and convolves it with the kernel
        if kernel.shape == region.shape:
            pooledvalue = np.sum(region*kernel)
            output[i//(x), j//(y)] = pooledvalue

print(output) # for debugging

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(array, cmap='gray') # show image  
axs[1].imshow(output, cmap='gray') # show block-convolved image
plt.show()

        
