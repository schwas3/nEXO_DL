import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img_as_img = np.load('sens_npy/gamma_166.root_184.npy')
for i in range(200):
    #do not plot channel without signal.
    if np.max(img_as_img[i, :, 1]) > 30:
        sample = []
        for j in range(255):
            if img_as_img[i,254 - j,1] == 0:
                sample.append(25)
            else:
                sample.append(img_as_img[i,254 - j,1])
        plt.plot(np.arange(255), np.array(sample) + 80*i, color='gray')
        plt.xlim(170, 255)
        #plt.ylim(4600, 5000)
plt.xlabel('Time index in the input array')
plt.ylabel('Amplitude + offset [a.u.]')
plt.savefig('sig_y.pdf')
#print(img_as_img.shape)
#npimg = np.array(img_as_img)
