import DCT
import os
import main
import logging
import PNG as EC
import cv2

ENCODE_SUBFOLDER = 'encoded'
DECODE_SUBFOLDER = 'decoded'
SLASH = '/'

class SequenceCoDec(DCT.CoDec):

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
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                encoded = self.encode_img(file)
                cv2.imwrite(ENCODE_SUBFOLDER + SLASH + file, encoded)
                logging.info(f'Encoded {file}')
        return 

    def decode(self):
        os.chdir(self.args.input_folder)
        encoded_folder = os.getcwd() + SLASH + ENCODE_SUBFOLDER
        logging.info(f'Looking for encoded images in directory: {encoded_folder}')
        files = os.listdir(encoded_folder)
        decode_folder = os.getcwd() + SLASH + DECODE_SUBFOLDER
        if not os.path.exists(decode_folder):
            os.mkdir(decode_folder)
        for file in files:
            if file.endswith('.png') or file.endswith('.jpg'):
                decoded = self.decode_img(encoded_folder + SLASH + file)
                cv2.imwrite(DECODE_SUBFOLDER + SLASH + file, decoded)
        return 

if __name__ == "__main__":
    parser = EC.parser
    parser.add_argument("-i", "--input_folder", type=str, help="Images input folder, overwrites encode and decode input", default="imgs")
    EC.parser_encode.add_argument("-b", "--block_size", type=int, help="Size fo the block of pixels used for tge DCT operation, if 4, a 4x4 block will be used", default=4)
    main.main(parser, logging, SequenceCoDec)