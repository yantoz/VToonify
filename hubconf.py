dependencies = [
    'torch', 'numpy', 
    'cv2', 'dlib', 'torchvision',
]

import os
import cv2
import torch
import bz2
import dlib
import tempfile
import numpy as np

from torchvision import transforms
import torch.nn.functional as F

from model.vtoonify import VToonify
from model.bisenet.model import BiSeNet
from model.encoder.align_all_parallel import align_face
from util import load_psp_standalone, get_video_crop_parameter, tensor2cv2


import logging

#logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger("vtoonify")

class _vToonify():
    
    def __init__(self, generator, exstyle, encoder, parsingpredictor, landmarkpredictor,
        style_degree=0.5, scale_image=True, padding=[200,200,200,200], color_transfer=True, device='cpu'):
        self._generator = generator
        self._exstyle = exstyle
        self._encoder = encoder
        self._parsingpredictor = parsingpredictor
        self._landmarkpredictor = landmarkpredictor
        self.style_degree = style_degree
        self.scale_image = scale_image
        self.padding = padding
        self.color_transfer = color_transfer
        self.device = device

    def __call__(self, image):

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
        ])
    
        scale = 1
        kernel_1d = np.array([[0.125],[0.375],[0.375],[0.125]])

        # We detect the face in the image, and resize the image so that the eye distance is 64 pixels.
        # Centered on the eyes, we crop the image to almost 400x400 (based on self.padding).
        if self.scale_image:
            paras = get_video_crop_parameter(image, self._landmarkpredictor, self.padding)
            if paras is not None:
                h, w, top, bottom, left, right, scale = paras
                H, W = int(bottom-top), int(right-left)
                # for HR image, we apply gaussian blur to it to avoid over-sharp stylization results
                if scale <= 0.75:
                    image = cv2.sepFilter2D(image, -1, kernel_1d, kernel_1d)
                if scale <= 0.375:
                    image = cv2.sepFilter2D(image, -1, kernel_1d, kernel_1d)
                image = cv2.resize(image, (w, h))[top:bottom, left:right]
    
        with torch.no_grad():
            I = align_face(image, self._landmarkpredictor)
            I = transform(I).unsqueeze(dim=0).to(self.device)
            s_w = self._encoder(I)
            s_w = self._generator.zplus2wplus(s_w)
            if self.color_transfer:
                s_w = self._exstyle
            else:
                s_w[:,:7] = self._exstyle[:,:7]
    
            x = transform(image).unsqueeze(dim=0).to(self.device)
            # parsing network works best on 512x512 images, so we predict parsing maps on upsmapled images
            # followed by downsampling the parsing maps
            x_p = F.interpolate(self._parsingpredictor(2*(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)))[0], 
                                scale_factor=0.5, recompute_scale_factor=False).detach()
            # we give parsing maps lower weight (1/16)
            inputs = torch.cat((x, x_p/16.), dim=1)
            log.debug("a = {}".format(x.shape))
            log.debug("b = {}".format(x_p.shape))
            
            y_tilde = self._generator(inputs, s_w.repeat(inputs.size(0), 1, 1), d_s = self.style_degree)        
            y_tilde = torch.clamp(y_tilde, -1, 1)

        output = ((y_tilde[0].detach().cpu().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
        return (image, output)

def vtoonify(
    progress=True, map_location=None,
    style="cartoon", style_id=26, style_degree=0.5,
    scale_image=True, padding=[200,200,200,200], color_transfer=True
):

    BASE_URL="https://huggingface.co/PKUWilliamYang/VToonify/resolve/main/models"

    if style == "caricature1":
        style = "caricature"
        ckpt = "vtoonify_s039_d0.5.pt"
    elif style == "caricature2":
        style = "caricature"
        ckpt = "vtoonify_s068_d0.5.pt"
    elif style == "illustration1":
        style = "illustration"
        ckpt = "vtoonify_s004_d_c.pt"
    elif style == "illustration2":
        style = "illustration"
        ckpt = "vtoonify_s009_d_c.pt"
    elif style == "illustration3":
        style = "illustration"
        ckpt = "vtoonify_s043_d_c.pt"
    elif style == "illustration4":
        style = "illustration"
        ckpt = "vtoonify_s054_d_c.pt"
    elif style == "illustration5":
        style = "illustration"
        ckpt = "vtoonify_s086_d_c.pt"
    else:
        ckpt = "vtoonify_s_d.pt"

    checkpoint_url="{}/vtoonify_d_{}/{}".format(BASE_URL, style, ckpt)


    exstyle_url="{}/vtoonify_d_{}/exstyle_code.npy".format(BASE_URL, style)
    encoder_url="{}/encoder.pt".format(BASE_URL)
    faceparsing_url="{}/faceparsing.pth".format(BASE_URL)    

    MODEL_DIR=os.path.join(torch.hub.get_dir(), "checkpoints", "vtoonify_d_{}".format(style))
    VTOONIFY_DIR=os.path.join(torch.hub.get_dir(), "checkpoints", "vtoonify")

    log.debug("Downloading {}".format(checkpoint_url))
    ckpt = torch.hub.load_state_dict_from_url(checkpoint_url, model_dir=MODEL_DIR, map_location=map_location, progress=progress)
    vtoonify = VToonify(backbone='dualstylegan')
    vtoonify.load_state_dict(ckpt['g_ema'])
    vtoonify.to(map_location)

    log.debug("Downloading {}".format(faceparsing_url))
    ckpt = torch.hub.load_state_dict_from_url(faceparsing_url, model_dir=VTOONIFY_DIR, map_location=map_location, progress=progress)
    parsingpredictor = BiSeNet(n_classes=19)
    parsingpredictor.load_state_dict(ckpt)
    parsingpredictor.to(map_location).eval()

    log.debug("Downloading {}".format(encoder_url))
    os.makedirs(VTOONIFY_DIR, exist_ok=True)
    DST = os.path.join(VTOONIFY_DIR, "encoder.pt")
    if not os.path.isfile(DST):
        torch.hub.download_url_to_file(encoder_url, DST, hash_prefix=None, progress=progress)
    pspencoder = load_psp_standalone(DST, map_location)    
     
    log.debug("Downloading {}".format(exstyle_url))
    os.makedirs(MODEL_DIR, exist_ok=True)
    DST = os.path.join(MODEL_DIR, "exstyle_code.npy")
    if not os.path.isfile(DST):
        torch.hub.download_url_to_file(exstyle_url, DST, hash_prefix=None, progress=progress)
    exstyles = np.load(DST, allow_pickle='TRUE').item()
    stylename = list(exstyles.keys())[style_id]
    exstyle = torch.tensor(exstyles[stylename]).to(map_location)
    with torch.no_grad():  
        exstyle = vtoonify.zplus2wplus(exstyle)
    
    MODEL = "shape_predictor_68_face_landmarks.dat"
    DST = os.path.join(VTOONIFY_DIR, MODEL)
    if not os.path.isfile(DST):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "{}.bz2".format(MODEL))
            url = "http://dlib.net/files/{}.bz2".format(MODEL)
            print("Downloading {} to {}".format(url, path)) 
            torch.hub.download_url_to_file(url, path, progress=progress)
            with bz2.BZ2File(path, 'r') as zip:
                data = zip.read()
                open(DST, 'wb').write(data)
    landmarkpredictor = dlib.shape_predictor(DST)

    model = _vToonify(vtoonify, exstyle, pspencoder, parsingpredictor, landmarkpredictor,
        style_degree=style_degree, scale_image=scale_image, padding=padding, color_transfer=color_transfer,
        device=map_location)
    return model

