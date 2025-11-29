import numpy as np
import pickle

def loadcif(f):
    with open(f, 'rb') as fileobj:
        d = pickle.load(fileobj, encoding='bytes')
    return d
batches = loadcif('cifar-10/data_batch_1'), loadcif('cifar-10/data_batch_2'), loadcif('cifar-10/data_batch_3'), loadcif('cifar-10/data_batch_4'), loadcif('cifar-10/data_batch_5')

all_images = [i[b'data'] for i in batches]
all_labels = [i[b'labels'] for i in batches]

egimg = all_images[0][23].reshape(3,32,32)/255
eglbl = all_labels[0][23]

class cnn:
    def __init__(self):
        self.kernels = {
            1: np.random.randn(8, 3, 5, 5) * np.sqrt(1/(5*5)),
            2: np.random.randn(16, 8, 3, 3) * np.sqrt(1/(3*3)),
            3: np.random.randn(32, 16, 3, 3) * np.sqrt(1/(2*2))}
        self.bias = {
            1: np.zeros(8),
            2: np.zeros(16),
            3: np.zeros(32)}

        self.w1 = np.random.randn(32, 16)*0.032
        self.b1 = np.zeros(16)

        self.w2 = np.random.randn(16, 10)*0.016
        self.b2 = np.zeros(10)

    def conv2d(self, inp, step, stride=1):
        k = self.kernels.get(step)
        b = self.bias.get(step)
        maps, __, ki, kj = k.shape
        __, inprows, inpcols = inp.shape
        outrows, outcols = int((inprows-ki)/stride + 1), int((inpcols-kj)/stride + 1)
        out = np.zeros((maps, outrows, outcols))

        for a in range(maps):
            for i in range(outrows):
                for j in range(outcols):
                    region = inp[:, i:i+ki, j:j+kj]
                    if region.shape == k[a].shape:
                        out[a, i, j] = np.sum(region * k[a]) + b[a]
        # if step==3:
        #     print(out.shape)
        return out

    def convpose2d(self, inp, kstep, amap):
        k = self.kernels.get(kstep) # 32, 16, 3, 3
        ai, ci, __, __ = k.shape
        __, ii, ji = inp.shape

        dinp = np.zeros_like(amap) 

        for a in range(ai):
            for i in range(ii):
                for j in range(ji):
                    for c in range(ci):
                        if dinp.shape == (3,3):
                            dinp[c, i:i+3, j:j+3] += inp[a, i, j]*k[a, c]
        
        return dinp

    def getkandbgrad(self, inp, step, cmap):
        bias = np.sum(inp, axis=(1,2))
        kd = np.zeros_like(self.kernels.get(step))

        ai, ii, ji = inp.shape

        for a in range(ai):
            for i in range(ii):
                for j in range(ji):
                    region = cmap[:, i:i+3, j:j+3]
                    "FIND THE ACTUAL IMAGE INPUTTED INTO KERNEL"
                    if region.shape == kd[a].shape:
                        kd[a] += region * inp[a, i, j] 
        return kd, bias

    def leakyrelu(self, inp):
        fmap = np.zeros_like(inp)
        locations = np.zeros_like(inp)
        aval, ival, jval = inp.shape

        for a in range(aval):
            for i in range(ival):
                for j in range(jval):
                    if inp[a, i, j]>0.0:
                        fmap[a, i, j] = inp[a, i, j]
                        locations[a, i, j] = 1
                    else:
                        fmap[a, i, j] = inp[a, i, j]*-0.01
                        locations[a, i, j] = -0.01
        return fmap, locations
    
    def pooler(self, inp, stride=2):
        aval, ival, jval = inp.shape 
        pooled = np.zeros((aval, int(ival/2), int(jval/2)))
        locations = np.zeros_like(inp)

        for a in range(aval):
            for i in range(0, ival, stride):
                for j in range(0, jval, stride):
                        region = inp[a, i:i+stride, j:j+stride]
                        if region.shape == (2,2):
                            pooled[a, i//2, j//2] = np.max(region) # divide i & j by two ---> when i=28, you want to put that at position 14, as there is no pos 28 in the pooled stuff
                            maxi, maxj = np.unravel_index(np.argmax(region), (2,2))
                            locations[a, i+maxi, j+maxj] = 1

        return pooled, locations

    def feedforward(self, inp):
        gapped = inp.mean(axis=(1, 2)) # means each 12x12, gap.shape = (128,)
        z1 = np.dot(gapped, self.w1) + self.b1 

        a = np.maximum(0, z1)
        z2 = np.dot(a, self.w2) + self.b2

        stabalized = z2-max(z2)

        y = np.exp(stabalized) / np.sum(np.exp(stabalized))

        return y, a, z1, gapped

    def run(self, inp):
        
        convmaps = [inp]
        activmaps = []
        leakylocations = []

        for i in range(3):
            convmap = self.conv2d(convmaps[i], i+1)
            activmap, leakylocation = self.leakyrelu(convmap)
            
            convmaps.append(convmap)
            activmaps.append(activmap)
            leakylocations.append(leakylocation)

        
        pooled, locations = self.pooler(activmaps[2])

        y, a, z1, gapped = self.feedforward(pooled)

        return y, a, z1, gapped, pooled, locations, leakylocations, activmaps, convmaps

    def getkernelderivs(self, poolderiv, leakylocations, activmaps, convmaps):

        """STEPS: 
            1. multiply leakylocations[step] by the upstream
            2. get kernel derivs, kernel biases
            3. unconvolve the upstream with the kernel
            4. set 3 to be the new upstream and repeat. should be 4 upstreams when done"""

        up = [poolderiv,]
        kderivs = []
        kbiases = []

        for i in range(3):
            workinup = leakylocations[2-i]*up[i]
            kd, kb = self.getkandbgrad(workinup, (3-i), convmaps[2-i])
            newup = self.convpose2d(workinup, (3-i), convmaps[2-i])
            
            kderivs.append(kd)
            kbiases.append(kb)
            up.append(newup)

        return kderivs, kbiases
        

    def backprop(self, lbl, img):
        yhat = np.eye(10)[lbl]
        y, ngac, ngz1, gapped, pooled, locations, leakylocations, activmaps, convmaps = self.run(img) # ngz1 = non-gradient z1, ngac = non gradient activation
        cce = -np.sum(yhat*np.log(y))

        ccesoftderiv = y - yhat
        w2 = np.outer(ngac,ccesoftderiv)
        b2 = 1*ccesoftderiv
        z2 = np.dot(self.w2, ccesoftderiv)

        w1 = np.outer(gapped, (ngz1>0)*(np.dot(w2, ccesoftderiv)))
        b1 = np.multiply((ngz1>0)/1, (np.dot(w2, ccesoftderiv)))
        z1 = np.dot(self.w1, z2)

        convupstream = np.zeros_like(pooled)
        ai, ii, ji = convupstream.shape

        for a in range(ai):
            convupstream[a, :, :] = np.full((ii,ji), (z1*144)[a])
        
        poolderiv = np.zeros_like(locations)
        ai, ii, ji = locations.shape

        for a in range(ai):
            for i in range(0, ii, 2):
                for j in range(0, ji, 2):
                    region = locations[a, i:i+2, j:j+2]
                    if region.shape == (2,2):
                        maxi, maxj = np.unravel_index(np.argmax(region), (2,2))
                        poolderiv[a, i+maxi, j+maxj] = convupstream[a, i//2, j//2]

        
        kderivs, kbiases = self.getkernelderivs(poolderiv, leakylocations, activmaps, convmaps)

        return cce, w2, b2, w1, b1, kderivs, kbiases, np.argmax(y)
    
    def updatevars(self, w2, b2, w1, b1, kderivs, kbiases, lr=0.1):
        self.w1 -= w1*lr
        self.b1 -= b1*lr

        self.w2 -= w2*lr
        self.b2 -= b2*lr

        for i in range(3):
            self.kernels[i+1] -= kderivs[2-i]
            self.bias[i+1] -= kbiases[2-i]

    def train(self, all_labels, all_images):
        # print("Beginning...")c
        for epoch in range(len(all_images)):
            total = 0
            correct = 0

            for i in range(len(all_images[epoch])):
                total+=1
                lbl = all_labels[epoch][i]
                img = all_images[epoch][i].reshape(3,32,32)/255
                cce, w2, b2, w1, b1, kderivs, kbiases, pred = self.backprop(lbl, img)
                if pred == lbl: 
                    correct+=1
                lr = 0.1 if i < 100 else 0.05 if i > 100 and i < 300 else 0.01
                self.updatevars(w2, b2, w1, b1, kderivs, kbiases, lr) 

                print(f"\r epoch: {epoch}, item: {i}, correct/total: {(correct/total)*100:.2f}", end="")

    def save(self):
        np.savez('cnn_params.npz',
            w1 = self.w1,
            b1 = self.b1, 

            w2 = self.w2, 
            b2 = self.b2,

            kernels = self.kernels, 
            biases = self.bias, 
        )
        print("Params saved.")
        

cnn = cnn()
# cnn.run(egimg)
# cnn.backprop(eglbl, egimg)

# mapses = cnn.conv2d(egimg, 1)
# lrelud = cnn.leakyrelu(mapses)


# # fig, axs = plt.subplots(4, 8, figsize=(10, 10))

# # for i in range(len(lrelud)):
# #     ax = axs[i // 8, i % 8]
# #     ax.imshow(mapses[i])
# #     ax.axis('off')

# # plt.tight_layout()
# # plt.show()

# mapses2 = cnn.conv2d(lrelud, 2)
# lrelud2 = cnn.leakyrelu(mapses2)

# mapses3 = cnn.conv2d(lrelud2, 3)
# relud = cnn.leakyrelu(mapses3)

# pooled, locations = cnn.pooler(relud)

# pred = cnn.feedforward(pooled)

# fig, axs = plt.subplots(8, 8, figsize=(10, 10))

# for i in range(len(mapses)):
#     ax = axs[i // 8, i % 8]
#     ax.imshow(mapses[i], cmap='gray')
#     ax.axis('off')

# plt.tight_layout()
# plt.show()


cnn.train(all_labels, all_images)

input("Hit enter to save.")

cnn.save()
