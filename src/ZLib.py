'''Entropy Encoding of images using PNG (Portable Network Graphics).'''

import argparse
import os
from skimage import io # pip install scikit-image
import logging
import zlib
import PIL.Image as Image
import io as readIO
FORMAT = "(%(levelname)s) %(module)s: %(message)s"
logging.basicConfig(format=FORMAT, level=logging.INFO)
#logging.basicConfig(format=FORMAT, level=logging.DEBUG)

import subprocess

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

def encode(codec):
    return codec.encode()

def decode(codec):
    return codec.decode()

ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"
ENCODE_OUTPUT = "encoded.bin"
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "decoded.png"

# Main parameters parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers(help="You must specify one of the following subcomands:", dest="subparser_name")

# Encoder parser
parser_encode = subparsers.add_parser("encode", help="Encode an image")
parser_encode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser_encode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")
parser_encode.set_defaults(func=encode)

# Decoder parser
parser_decode = subparsers.add_parser("decode", help='Decode an image')
parser_decode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser_decode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {DECODE_OUTPUT}", default=f"{DECODE_OUTPUT}")    
parser_decode.set_defaults(func=decode)

class CoDec:

    def __init__(self, args):
        self.args = args
        logging.debug(f"args = {self.args}")
        if args.subparser_name == "encode":
            self.encoding = True
        else:
            self.encoding = False
        logging.debug(f"encoding = {self.encoding}")
        self.required_bytes = 0

    def encode(self):
        '''Read an image and save it in the disk.'''
        img = self.read(write_local=True)
        imagen = open(self.args.input, 'rb').read()
        compressed_data = zlib.compress(imagen, zlib.Z_BEST_COMPRESSION)
        with open(self.args.output, "wb") as out_file:
            out_file.write(compressed_data)
        wbytes= os.path.getsize(self.args.input)*8
        os.remove(self.args.input)
        return wbytes/(img.shape[0]*img.shape[1])

    def decode(self):
        imagen = open(self.args.input, 'rb').read()
        decompressed = zlib.decompress(imagen)
        img = Image.open(readIO.BytesIO(decompressed))
        img.save(self.args.output)
        os.remove(self.args.input)
        self.required_bytes = os.path.getsize(self.args.output)
        logging.info(f"Written {self.required_bytes} bytes in {self.args.output}")
        return self.required_bytes*8/(img.size[0]*img.size[1])

    def read(self, write_local = False):
        img = io.imread(self.args.input)
        if write_local:
            io.imsave("temp.png", img)
            logging.info(f"Read {self.args.input} of shape {img.shape}")
            self.args.input = "temp.png"
        return img

    def read_fn(self, fn):
        '''Read an image.'''
        img = io.imread(fn)
        logging.info(f"Read {fn} of shape {img.shape}")
        return img

    def save_fn(self, img, fn):
        '''Save to disk the image with filename <fn>.'''
        # The encoding algorithm depends on the output file extension.
        io.imsave(fn, img, check_contrast=False)
        subprocess.run(f"optipng {fn}", shell=True, capture_output=True)
        self.required_bytes = os.path.getsize(fn)
        logging.info(f"Written {self.required_bytes} bytes in {fn}")

    def save(self, img):
        self.save_fn(img, self.args.output)
        
if __name__ == "__main__":
    logging.info(__doc__) # ?
    parser.description = __doc__
    #args = parser.parse_known_args()[0]
    args = parser.parse_args()

    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("You must specify 'encode' or 'decode'")
        quit()

    codec = CoDec(args)
    rate = args.func(codec)
    logging.info(f"rate = {rate} bits/pixel")