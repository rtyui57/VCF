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
            self.block_size = args.block_size
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
        dct = self.analyze_image(YCoCg, self.block_size, self.block_size)
        if self.block_size != 8:
            dct = self.quantize(dct).astype(np.float32)
        self.write_float64(dct)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def encode_img(self, img_path, bloc):
        self.block_size = bloc
        img = cv2.imread(img_path)
        YCoCg = from_RGB(img)
        dct = self.analyze_image(YCoCg, bloc, bloc)
        dct = self.quantize(dct).astype(np.float32)
        return dct

    def decode(self):
        k = self.read_float64()
        if self.block_size != 8:
            k = self.dequantize(k)
        dct = self.synthesize_image(k, self.block_size, self.block_size).astype(np.uint8)
        YCoCg = to_RGB(dct)
        self.write(YCoCg)
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate

    def decode_img(self, img_path, bloc):
        self.block_size = bloc
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        k = self.dequantize(img)
        dct = self.synthesize_image(k, bloc, bloc).astype(np.uint8)
        YCoCg = to_RGB(dct)
        return YCoCg
    
    def apply_quantize(self, block):
        result = np.copy(block)
        quantization_table_lum = self.get_cl_table()
        quantization_table_chrm = self.get_cr_table()
        for i in range (0, 8):
            for j in range (0, 8):
                result[i][j][0] = self.quantize_scalar(block[i][j][0], quantization_table_lum[i][j])
                result[i][j][1] = self.quantize_scalar(block[i][j][1], quantization_table_chrm[i][j])
                result[i][j][2] = self.quantize_scalar(block[i][j][2], quantization_table_chrm[i][j])
        return result


    def apply_dequantize(self, block):
        result = np.copy(block)
        quantization_table_lum = self.get_cl_table()
        quantization_table_chrm = self.get_cr_table()
        for i in range (0, 8):
            for j in range (0, 8):
                result[i][j][0] = self.dequantize_scalar(block[i][j][0], quantization_table_lum[i][j])
                result[i][j][1] = self.dequantize_scalar(block[i][j][1], quantization_table_chrm[i][j])
                result[i][j][2] = self.dequantize_scalar(block[i][j][2], quantization_table_chrm[i][j])
        return result

    def quantize_scalar(self, x, steps):
        k = (x / steps).astype(np.int8)
        k += 128
        return k.astype(np.uint8)
    
    def dequantize_scalar(self, x, steps):
        x = x.astype(np.int16) - 128
        y = np.where(x < 0, steps * (x - 0.5), x)
        y = np.where(x > 0, steps * (x + 0.5), y)
        return y

    def analyze_image(self, image, block_y_side, block_x_side):
        '''DCT image transform by blocks.'''
        blocks_in_y = image.shape[0]//block_y_side
        blocks_in_x = image.shape[1]//block_x_side
        image_DCT = np.empty_like(image, dtype=np.float32)
        for y in range(blocks_in_y):
            for x in range(blocks_in_x):
                block = image[y*block_y_side:(y+1)*block_y_side,
                            x*block_x_side:(x+1)*block_x_side]
                DCT_block = block_DCT.analyze_block(block)
                if self.block_size == 8:
                    DCT_block = self.apply_quantize(DCT_block)
                image_DCT[y*block_y_side:(y+1)*block_y_side,
                        x*block_x_side:(x+1)*block_x_side] = DCT_block
        return image_DCT
    
    def synthesize_image(self, image_DCT, block_y_side, block_x_side):
        '''Inverse DCT image transform by blocks.'''
        blocks_in_y = image_DCT.shape[0]//block_y_side
        blocks_in_x = image_DCT.shape[1]//block_x_side
        #image = np.empty_like(image_DCT, dtype=np.int16) # Ojo!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        image = np.empty_like(image_DCT)
        for y in range(blocks_in_y):
            for x in range(blocks_in_x):
                DCT_block = image_DCT[y*block_y_side:(y+1)*block_y_side,
                                    x*block_x_side:(x+1)*block_x_side]
                if self.block_size == 8:
                    DCT_block = self.apply_dequantize(DCT_block)
                block = block_DCT.synthesize_block(DCT_block)
                image[y*block_y_side:(y+1)*block_y_side,
                    x*block_x_side:(x+1)*block_x_side] = block
        return image

    def get_cl_table(self):
        return [[16, 11, 10, 16, 24, 40, 51, 61], 
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]]

    def get_cr_table(self):
        return [[17, 18, 24, 47, 99, 99, 99, 99], 
                [18, 21, 26, 66, 99, 99, 99, 99],
                [24, 26, 56, 99, 99, 99, 99, 99],
                [47, 66, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99],
                [99, 99, 99, 99, 99, 99, 99, 99]]

    def write_float64(self, img):
        cv2.imwrite(self.args.output.replace('.png', '.exr'), img)

    def read_float64(self):
        return cv2.imread(self.args.input.replace('.png', '.exr'), cv2.IMREAD_UNCHANGED)


if __name__ == "__main__":
    parser = EC.parser_encode
    parser.add_argument("-b", "--block_size", type=int, help="Size fo the block of pixels used for tge DCT operation, if 4, a 4x4 block willbe used", default=8)
    main.main(EC.parser, logging, CoDec)