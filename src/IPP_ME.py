import LloydMax as deadzone
import os
import main
import logging
import PNG as EC
import cv2 
import numpy as np
from motion_estimation import full_search, prediction
from scipy import ndimage
import logging

ENCODE_SUBFOLDER = 'encoded'
DECODE_SUBFOLDER = 'decoded'
SLASH = '\\'

class SequenceCoDec(deadzone.CoDec):

    def __init__(self, args):
        super().__init__(args)

    def encode(self):
        os.chdir(self.args.input_folder)
        input_folder = os.getcwd()
        logging.info(f'Looking for images in directory: {input_folder}')
        files = os.listdir(input_folder)
        encode_folder = input_folder + SLASH + ENCODE_SUBFOLDER
        if not os.path.exists(encode_folder):
            os.mkdir(encode_folder)
        self._encode(files, input_folder)
        return 

    def decode(self):
        os.chdir(self.args.input_folder)
        encoded_folder = os.getcwd() + SLASH + ENCODE_SUBFOLDER
        logging.info(f'Looking for encoded images in directory: {encoded_folder}')
        files = os.listdir(encoded_folder)
        decode_folder = os.getcwd() + SLASH + DECODE_SUBFOLDER
        if not os.path.exists(decode_folder):
            os.mkdir(decode_folder)
        self._decode(files, encoded_folder)
        return 

    def _encode(self, files, input_folder):
        current = None
        next = None
        index = 0
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                if index == 0:
                    self.args.input = input_folder + SLASH + file
                    self.args.output = ENCODE_SUBFOLDER + SLASH + str(index) + '.png'
                    super().encode()
                    current = cv2.imread(file)
                else:
                    next = cv2.imread(file)
                    MVs = full_search.block_ME(next, current, 32, 8)
                    _MVs = np.empty((next.shape[0], next.shape[1], 3), dtype=np.float32)
                    _MVs[..., 0] = ndimage.zoom(MVs[..., 0], next.shape[0]/MVs.shape[0], order=0)
                    _MVs[..., 1] = ndimage.zoom(MVs[..., 1], next.shape[0]/MVs.shape[0], order=0)
                    predicted = prediction.make(current,_MVs[:,:, 0:1])
                    path = ENCODE_SUBFOLDER + SLASH + str(index) + '.png'
                    cv2.imwrite(ENCODE_SUBFOLDER + SLASH + str(index) + 'VM.png', _MVs)
                    cv2.imwrite(path, self.resta(predicted, next))
                    self.args.input = path
                    self.args.output = path
                    super().encode()
                    current = cv2.imread(file)
                index += 1
        logging.info(f'{index} images have been encoded')
    
    def _decode(self, files, encoded_folder):
        current = None
        index = 0
        for file in files:
            if file.endswith('.png') and not file.endswith('VM.png'):
                if index == 0:
                    self.args.input = encoded_folder + SLASH + file
                    self.args.output = DECODE_SUBFOLDER + SLASH + file
                    super().decode()
                    current = cv2.imread(DECODE_SUBFOLDER + SLASH + str(index) + '.png')
                else:
                    path = DECODE_SUBFOLDER + SLASH + str(index)  + '.png'
                    self.args.input = encoded_folder + SLASH + file
                    self.args.output = path
                    print(self.args.input)
                    super().decode()
                    difference = cv2.imread(path)
                    _MVs = cv2.imread(encoded_folder + SLASH + str(index) + 'VM.png')
                    predicted = prediction.make(current,_MVs[:,:, 0:1])
                    next = self.suma(predicted, difference)
                    cv2.imwrite(path, next)
                    current = next
                index += 1
        return

    def resta(self, i1, i2):
        return np.clip(i2.astype(np.int16) - i1 + 128, 0, 255).astype(np.uint8)

    def suma(self, i1, restos):
        return np.clip(i1.astype(np.int16) + restos - 128, 0, 255).astype(np.uint8)

if __name__ == "__main__":
    EC.parser.add_argument("-f", "--input_folder", type=str, help="Images input folder, overwrites encode and decode input", default="imgs")
    main.main(EC.parser, logging, SequenceCoDec)