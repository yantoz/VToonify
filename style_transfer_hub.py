import os
import argparse

import logging

#logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("main")


class TestOptions():
    def __init__(self):

        self.parser = argparse.ArgumentParser(description="Style Transfer")
        self.parser.add_argument("--content", type=str, default='./data/077436.jpg', help="path of the content image")
        self.parser.add_argument("--style_id", type=int, default=26, help="the id of the style image")
        self.parser.add_argument("--style_degree", type=float, default=0.5, help="style degree for VToonify-D")
        self.parser.add_argument("--color_transfer", action="store_true", help="transfer the color of the style")
        self.parser.add_argument("--ckpt", type=str, default='cartoon', help="path of the saved model")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--scale_image", action="store_true", help="resize and crop the image to best fit the model")
        self.parser.add_argument("--cpu", action="store_true", help="if true, only use cpu")
        self.parser.add_argument("--padding", type=int, nargs=4, default=[200,200,200,200], help="left, right, top, bottom paddings to the face center")
        
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        log.debug('Load options')
        for name, value in sorted(args.items()):
            log.debug("{}: {}".format(name, value))
        return self.opt
   
parser = TestOptions()
args = parser.parse()
log.debug('*'*98)
      
device = "cpu" if args.cpu else "cuda"

if args.cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

import cv2
import torch 
import numpy as np

if __name__ == "__main__":

    parser = TestOptions()
    args = parser.parse()
    log.debug('*'*98)
    
    device = "cpu" if args.cpu else "cuda"

    if args.cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ""
     
    model = torch.hub.load('.', 'vtoonify', source='local', map_location='cpu',
        style=args.ckpt, style_id=args.style_id, style_degree=args.style_degree, 
        scale_image=args.scale_image, padding=args.padding, color_transfer=args.color_transfer)

    filename = args.content
    basename = os.path.basename(filename).split('.')[0]
    log.debug('Processing ' + os.path.basename(filename) + ' with vtoonify_d')
    
    cropname = os.path.join(args.output_path, basename + '_input.jpg')
    savename = os.path.join(args.output_path, basename + '_vtoonify_d.jpg')

    frame = cv2.imread(filename)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    log.debug("Input: {}".format(frame.shape))
    frame, output = model(frame)
    log.debug("Output: {}".format(output.shape))
    
    cv2.imwrite(cropname, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    cv2.imwrite(savename, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
        
    log.debug('Transfer style successfully!')
