"""
Test single image, get the attention map of a particular layer.
To visualize the attention map, maybe attention is meanless?
"""
import json
import models
import timm
import os
import torch
import argparse
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from einops import rearrange
from torch import einsum
from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models

object_categories = []
with open("imagenet1k_id_to_label.txt", "r") as f:
    for line in f:
        _, val = line.strip().split(":")
        object_categories.append(val)


parser = argparse.ArgumentParser(description='PyTorch ImageNet Single Image Testing')

parser.add_argument('--image', type=str, default="images/demoC.JPEG", help='path to image')
parser.add_argument('--shape', type=int, default=224, help='path to image')

# Model parameters
#"deit_base_patch16_224", "vit_large_patch16_224", "vit_base_patch16_224","cait_s24_224"]
parser.add_argument('--model', default='fcvt_v5_64_B12', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--stage', default=3, type=int,
                    help='Index of visualized stage, 0-3')
parser.add_argument('--block', default=-1, type=int,
                    help='Index of visualized stage, -1 is the last block')
parser.add_argument('--group', default=7, type=int,
                    help='Index of visualized attention head, 0-7')
parser.add_argument('--query', default=23, type=int,
                    help='Index of query patch, ranging from (0-195), 14x14')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

args = parser.parse_args()
# default setting of args.checkpoint
args.checkpoint = f"./images/out/{args.model}.tar"
assert args.model in timm.list_models(), "Please use a timm pre-trined model, see timm.list_models()"

# Preprocessing
def _preprocess(image_path):
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image


# forward hook function
def get_attention_score(self, input, output):
    # especially design for Vision Transformers.
    # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    # token number is 197, remove the first one, class token.
    x = input[0] # input tensor in a tuple

    b,c,w,h = x.size()
    x = rearrange(x,"b c x y -> b c (x y)")
    gap = x.mean(dim=-1, keepdim=True)
    q, g = map(lambda t: rearrange(t, 'b (h d) n -> b h d n', h = self.head), [x,gap])  #[b,head, hdim, n]
    sim = einsum('bhdi,bhjd->bhij', q, g.transpose(-1, -2)).squeeze(dim=-1) * self.scale  #[b,head, w*h]
    std, mean = torch.std_mean(sim, dim=[1,2], keepdim=True)
    sim = (sim-mean)/(std+self.epsilon)
    sim = sim * self.rescale_weight.unsqueeze(dim=0).unsqueeze(dim=-1) + self.rescale_bias.unsqueeze(dim=0).unsqueeze(dim=-1)
    sim = sim.reshape(b,self.head,1, w, h) # [b, head, 1, w, h]
    sim = sim.squeeze(dim=2)# [b, head, w, h]

    global attention
    attention = sim.detach()



def show_query_on_image(img_path, row, col, save_path, color=[0,0,255]):
    img = cv2.imread(img_path, 1)
    img = cv2.resize(img, (224, 224))
    img[row*16:(row+1)*16, col*16:(col+1)*16, 0] = color[0]
    img[row*16:(row+1)*16, col*16:(col+1)*16, 1] = color[1]
    img[row*16:(row+1)*16, col*16:(col+1)*16, 2] = color[2]
    cv2.imwrite(save_path, np.uint8(img))



def show_cam_on_image(img_path, mask, save_path, is_query=False):
    img = cv2.imread(img_path, 1)
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    if is_query:
        color_map = cv2.COLORMAP_MAGMA
        # mask = 1-mask
    else:
        color_map = cv2.COLORMAP_HOT
    heatmap = cv2.applyColorMap(np.uint8(255*mask), color_map)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite(save_path, np.uint8(255 * cam))


def main():
    global attention
    # predict the image
    image, raw_image = _preprocess(args.image)
    image = image.unsqueeze(dim=0)
    model = timm.create_model(model_name=args.model,pretrained=True)
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, True)
    model.network[args.stage*2][args.block].token_mixer.gc2.register_forward_hook(get_attention_score)
    out = model(image)
    if type(out) is tuple:
        out =out[0]
    possibility = torch.softmax(out,dim=1).max()
    value, index = torch.max(out, dim=1)
    print(f'Prediction is: {object_categories[index]} possibility: {possibility*100:.3f}%')

    try:
        os.makedirs(f"images/out/{args.model}/")
    except:
        pass
    image_name = os.path.basename(args.image).split(".")[0]
    # process the attention map
    attention = attention[0, args.group, :, :]
    mask = attention.unsqueeze(dim=0).unsqueeze(dim=0)
    mask = F.interpolate(mask,(224,224))
    mask = mask.squeeze(dim=0).permute(1,2,0)
    mask = (mask -mask.min())/(mask.max()-mask.min()) # normalize to 0-1
    show_cam_on_image(args.image, mask, f"images/out/{args.model}/{image_name}_S{args.stage}_B{args.block}_G{args.group}.png",  is_query=False)
    print(f"Atten image is saved to: images/out/{args.model}/{image_name}_S{args.stage}_B{args.block}_G{args.group}.png")
    # print(f"attention shape is {attention.shape}")







if __name__ == '__main__':
    main()
