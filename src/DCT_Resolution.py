'''Exploiting color (perceptual) redundancy with the YCoCg transform.'''
import numpy as np
import cv2
import logging
import main
import PNG as EC
import deadzone as Q
from color_transforms.YCrCb import from_RGB, to_RGB
from DCT2D import block_DCT

class CoDec(Q.CoDec):

    def __init__(self, args):
        super().__init__(args)
        if self.encoding:
            self.block_size = 2**args.block_size
            logging.info(f"Block size = {self.block_size}")
            with open("block_size.txt", 'w') as f:
                f.write(f"{self.block_size}")
                logging.info("Written block_size.txt")
        else:
            self.resolution = 2**args.resolution
            with open("block_size.txt", 'r') as f:
                self.block_size = int(f.read())
                logging.info(f"Read Block size = {self.block_size} from block_size.txt")

    def encode(self):
        img = self.read()
        YCoCg = from_RGB(img)
        #dct = block_DCT.analyze_image(YCoCg, self.block_size, self.block_size)
        subbands = block_DCT.get_subbands(YCoCg, self.block_size, self.block_size)
        quantized = self.quantize(subbands).astype(np.float32)
        self.write_float64(quantized)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        resolution_level = self.block_size//self.resolution
        k = self.read_float64()
        dequantized = self.dequantize(k)
        blocks = block_DCT.get_blocks(dequantized, self.block_size//resolution_level, self.block_size//resolution_level)
        #dct = block_DCT.synthesize_image(blocks, self.block_size, self.block_size).astype(np.uint8)
        YCoCg = to_RGB(blocks.astype(np.uint8))
        self.write(YCoCg[0:k.shape[0]//resolution_level,0:k.shape[1]//resolution_level,:])
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate

    def write_float64(self, img):
        cv2.imwrite(self.args.output.replace('.png', '.exr'), img)

    def read_float64(self):
        return cv2.imread(self.args.input.replace('.png', '.exr'), cv2.IMREAD_UNCHANGED)


if __name__ == "__main__":
    parser = EC.parser
    parser.add_argument("-b", "--block_size", type=int, help="Affects the size fo the block of pixels used for the DCT operation, it will be eleveated in two", default=3)
    EC.parser_decode.add_argument("-r", "--resolution", type=int, default=3)
    main.main(EC.parser, logging, CoDec)