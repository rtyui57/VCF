'''Exploiting color (perceptual) redundancy with the YCoCg transform.'''
import numpy as np
import cv2
import logging
import main
import PNG as EC
import deadzone as Q
from color_transforms.YCrCb import from_RGB, to_RGB
from DCT2D import block_DCT
import os
from information_theory import information, distortion

class CoDec(Q.CoDec):

    def __init__(self, args):
        super().__init__(args)
        if self.encoding:
            if args.automatic:
                self.block_size = self.get_block_size(args.lambdas)
            else:
                self.block_size = 2**args.block_size
            logging.info(f"Block size = {self.block_size}")
            with open("block_size.txt", 'w') as f:
                f.write(f"{self.block_size}")
                logging.info("Written block_size.txt")
        else:
            with open("block_size.txt", 'r') as f:
                self.block_size = int(f.read())
                logging.info(f"Read Block size = {self.block_size} from block_size.txt")

    def encode(self):
        img = self.read()
        YCoCg = from_RGB(img)
        dct = block_DCT.analyze_image(YCoCg, self.block_size, self.block_size)
        quantized = self.quantize(dct).astype(np.float32)
        self.write_float64(quantized)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        k = self.read_float64()
        dequantized = self.dequantize(k)
        dct = block_DCT.synthesize_image(dequantized, self.block_size, self.block_size).astype(np.uint8)
        YCoCg = to_RGB(dct.astype(np.uint8))
        self.write(YCoCg)
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate
    
    def get_block_size(self, lambdas):
        logging.info(f"Using {lambdas} as lamba in lagrangian optimization")
        min = 1000000
        result = 0
        for i in range(0, 7):
            block_size = 2**i
            #Encode
            img = self.read()
            YCoCg = from_RGB(img)
            dct = block_DCT.analyze_image(YCoCg, block_size, block_size)
            quantized = block_DCT.uniform_quantize(dct, block_size, block_size, 3, self.QSS)
            size = self.write_float64(quantized)
            
            #Decode
            dequantized = block_DCT.uniform_dequantize(quantized, block_size, block_size, 3, self.QSS)
            dct = block_DCT.synthesize_image(dequantized, block_size, block_size).astype(np.uint8)
            YCoCg = to_RGB(dct.astype(np.uint8))
            #Get lagraungian
            rate = (size*8)/(img.shape[0]*img.shape[1])
            RMSE = distortion.RMSE(img, YCoCg)
            J = rate + (lambdas*RMSE)
            logging.info("Block size of " + str(block_size) + " has J: " + str(J))
            if J < min:
                min = J
                result = block_size
        return result


    def write_float64(self, img):
        path = self.args.output#.replace('.png', '.exr')
        cv2.imwrite(path, img)
        return os.path.getsize(path)

    def read_float64(self):
        return cv2.imread(self.args.input, cv2.IMREAD_UNCHANGED)


if __name__ == "__main__":
    parser = EC.parser_encode
    parser.add_argument("-b", "--block_size", type=int, help="Affects the size fo the block of pixels used for the DCT operation, it will be eleveated in two", default=3)
    parser.add_argument("-a", "--automatic", type=bool, help="Detect the most optime block size", default=False)
    parser.add_argument("-l", "--lambdas", type=float, help="Lambda used in lagrangian optimization", default=1)
    main.main(EC.parser, logging, CoDec)