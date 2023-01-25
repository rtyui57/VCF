'''Exploiting color (perceptual) redundancy with the YCoCg transform.'''

import numpy as np
import logging
import main
import PNG as EC
import deadzone as Q
from color_transforms.YCrCb import from_RGB, to_RGB # pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"

class CoDec(Q.CoDec):

    def __init__(self, args):
        super().__init__(args)
        if self.encoding:
            self.redundancy = args.chroma_redundancy
            logging.info(f"Redundacy = {self.redundancy}")
            with open("redundacy.txt", 'w') as f:
                f.write(f"{self.redundancy}")
                logging.info("Written redundacy.txt")
        else:
            with open("redundacy.txt", 'r') as f:
                self.redundancy = int(f.read())
                logging.info(f"Read Redundancy={self.redundancy} from redundancy.txt")

    def encode(self):
        img = self.read()
        YCrCb_img = from_RGB(img)
        luma = YCrCb_img[:,:, 0]
        chroma = YCrCb_img[:,:, 1:]
        quantized = np.copy(img)
        logging.info(f"Using {self.QSS} as Quantization step for quantization of the luma")
        quantized[:,:,0] = self.quantize(luma)
        self.QSS = self.QSS * self.redundancy
        logging.info(f"Using {self.QSS} as Quantization step for quantization of the chroma")
        chroma = self.quantize(chroma)
        quantized[:,:,1:3] = chroma
        self.write(quantized)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        k = self.read()
        YCrCb_y = np.copy(k)
        logging.info(f"Using {self.QSS} as Quantization step for dequantization of the luma")
        YCrCb_y[:,:,0] = self.dequantize(k[:,:,0])
        self.QSS = self.QSS * self.redundancy
        logging.info(f"Using {self.QSS} as Quantization step for dequantization of the luma")
        YCrCb_y[:,:,1:3] = self.dequantize(k[:,:,1:3])
        #y_128 = to_RGB(YCrCb_y.astype(np.int16))
        y = to_RGB(YCrCb_y)
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.write(y)
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate

if __name__ == "__main__":
    parser = EC.parser
    parser.add_argument("-r", "--chroma_redundancy", type=int, help="Multiplies the quantization step for htis parameter to use in the chroma quantization", default=1)
    main.main(EC.parser, logging, CoDec)
