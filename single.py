"""
Test single image, get the attention map of a particular layer.
To visualize the attention map, maybe attention is meanless?
"""
import json

import timm
import os
import torch
import argparse
import cv2
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

object_categories = []
with open("imagenet1k_id_to_label.txt", "r") as f:
    for line in f:
        _, val = line.strip().split(":")
        object_categories.append(val)


parser = argparse.ArgumentParser(description='PyTorch ImageNet Single Image Testing')

parser.add_argument('--image', type=str, default="images/demo6.JPEG", help='path to image')
parser.add_argument('--shape', type=int, default=224, help='path to image')

# Model parameters
parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50"')
parser.add_argument('--layer', default=11, type=int,
                    help='Index of visualized block, 0-11')
parser.add_argument('--head', default=9, type=int,
                    help='Index of visualized attention head')
parser.add_argument('--query', default=143, type=int,
                    help='Index of query patch, ranging from (0-195), 14x14')

args = parser.parse_args()
assert args.model in timm.list_models(), "Please use a timm pre-trined model, see timm.list_models()"
assert args.model in ["deit_base_distilled_patch16_224", "deit_base_patch16_224", "vit_large_patch16_224", "vit_base_patch16_224"], "Currently, only support Vision Transformers patch 16, size 224"
# vit_base_patch16_224  double-check.

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
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(dim=-1)
    global attention
    attention = attn.detach()



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
    model.blocks[args.layer].attn.register_forward_hook(get_attention_score)
    out = model(image)
    if type(out) is tuple:
        out =out[0]
    possibility = torch.softmax(out,dim=1).max()
    value, index = torch.max(out, dim=1)
    print(f'Prediction is: {object_categories[index]} possibility: {possibility*100:.3f}%')



    image_name = os.path.basename(args.image).split(".")[0]
    # save query image
    mask = torch.zeros(1,1,14,14)
    row = int(args.query/14)
    col = args.query%14
    mask[0, 0, row, col] = 1.0
    mask = F.interpolate(mask,(224,224))
    mask = mask.squeeze(dim=0).permute(1,2,0)
    try:
        os.makedirs(f"images/out/{args.model}/")
    except:
        pass
    show_cam_on_image(args.image, mask, f"images/out/{args.model}/{image_name}_Q{args.query}.png",  is_query=True)
    print(f"Query image is saved to: images/out/{args.model}/{image_name}_Q{args.query}.png")
    # cv2.imwrite("images/out/save.png", raw_image)



    # process the attention map [tokens, tokens], remove the class_token (first token)
    attention = attention[0, args.head, 1:, 1:]
    mask = (attention[args.query,:]).reshape(14,14).unsqueeze(dim=0).unsqueeze(dim=0)
    mask = F.interpolate(mask,(224,224))
    mask = mask.squeeze(dim=0).permute(1,2,0)
    mask = (mask -mask.min())/(mask.max()-mask.min()) # normalize to 0-1
    show_cam_on_image(args.image, mask, f"images/out/{args.model}/{image_name}_Q{args.query}_L{args.layer}_H{args.head}.png",  is_query=False)
    print(f"Atten image is saved to: images/out/{args.model}/{image_name}_Q{args.query}_L{args.layer}_H{args.head}.png")
    # print(f"attention shape is {attention.shape}")







if __name__ == '__main__':
    main()
