import scipy.misc
import numpy as np
import math

def load_image(file,size):
    return scipy.misc.imresize(scipy.misc.imread(file).astype(np.float),[size,size,3])/127.5 - 1

def save_mosaic(images,file):
        count = images.shape[0]
        output_image = np.zeros((64 * 8,64 * int(math.ceil(count/8.0)),3))
        for i in range(count):
            x = i%8
            y = i//8
            output_image[x*64:(x+1)*64,y*64:(y+1)*64,:] = images[i]

        image = scipy.misc.toimage(output_image,cmin=-1,cmax=1)
        image.save(file);
