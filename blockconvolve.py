import numpy as np
import matplotlib.pyplot as plt


# load image
array = np.load('array.npy')

kernel = np.array([[1/16, 1/16, 1/16, 1/16], 
                   [1/16, 1/16, 1/16, 1/16],
                   [1/16, 1/16, 1/16, 1/16],
                   [1/16, 1/16, 1/16, 1/16]])

output = np.zeros((8, 8))

# block convultion; image is 32x32, kernel is 4x4 and moves over 4 px every time to ensure no overlap

for i in range(0, 32, 4):
    for j in range(0, 32, 4):
        region = array[i:i+4, j:j+4] # slices a 4x4 section of the the array and convolves it with the kernel
        if len(kernel) == len(region):
            multiply = np.dot(region, kernel)
            pooledvalue = np.sum(multiply)/4
            output[i//4, j//4] = pooledvalue

print(output) # for debugging

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].imshow(array, cmap='gray') # show image  
axs[1].imshow(output, cmap='gray') # show block-convolved image
plt.show()

        
