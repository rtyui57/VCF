'''Exploiting color (perceptual) redundancy with the YCoCg transform.'''
import numpy as np
import logging
import main
import PNG as EC
import deadzone as Q
from color_transforms.DCT import from_RGB, to_RGB

class CoDec(Q.CoDec):

    def encode(self):
        img = self.read()
        DCT_img = from_RGB(img)
        k = self.quantize(DCT_img)
        self.write(k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        k = self.read()
        DCT = self.dequantize(k)
        #y_128 = to_RGB(YCoCg_y.astype(np.int16))
        y = to_RGB(DCT)
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.write(y)
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate

if __name__ == "__main__":
    main.main(EC.parser, logging, CoDec)