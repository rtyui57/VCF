'''Exploiting color (perceptual) redundancy with the YCoCg transform.'''
import numpy as np
import logging
import main
import PNG as EC
import deadzone as Q
from color_transforms.DCT import from_RGB, to_RGB
import cv2 
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

    def decode_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        k = self.dequantize(img)
        YCoCg = to_RGB(k)
        return YCoCg

    def encode_img(self, img_path):
        img = cv2.imread(img_path)
        YCoCg = from_RGB(img)
        dct = self.quantize(YCoCg)
        return dct

if __name__ == "__main__":
    main.main(EC.parser, logging, CoDec)