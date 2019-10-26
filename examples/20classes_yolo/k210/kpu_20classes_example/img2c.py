import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == '__main__':
    img = plt.imread(sys.argv[1])
    img = np.transpose(img, [2,0,1])
    with open('image.c', 'w') as f:
        print('const unsigned char gImage_image[]  __attribute__((aligned(128))) ={', file=f)
        print(', '.join([str(i) for i in img.flatten()]), file=f)
        print('};', file=f)