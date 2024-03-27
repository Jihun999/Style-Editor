import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
from pycocotools import mask as maskUtils
import cv2
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import random
import clip
from utils import utils
from tqdm import tqdm

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torch

def get_features(image, VGG):
    layers = {
        '6': 'relu2_1',
        '11': 'relu3_1',
        '20': 'relu4_1',
        '29': 'relu5_1'
    }
    features = {}
    x = image.float()
    for name, layer in VGG._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    return features


def vgg_loss(content_img, target_img, VGG, device):
    VGG = VGG.to(device)
    content_img = content_img.unsqueeze(0).to(device)
    target_img = target_img.unsqueeze(0).to(device)
    content_img = get_features(content_img, VGG)
    target_img = get_features(target_img, VGG)

    # content loss
    content_loss = 0
    mse = torch.nn.MSELoss(reduction='mean')
    w = [1/len(content_img.keys())] * len(content_img.keys())
    for idx, i in enumerate(content_img.keys()):
        content_loss += mse(content_img[i], target_img[i])
        # content_loss += torch.norm(content_img[i] - target_img[i], p=2) / (content_img[i].shape[1] * content_img[i].shape[2] * content_img[i].shape[3])
    
    # style loss
    style_loss = 0
    for i in content_img.keys():
        mean_style = mse(content_img[i].mean(dim=1), target_img[i].mean(dim=1))
        std_style = mse(content_img[i].std(dim=1), target_img[i].std(dim=1))
        # mena_style = mse(torch.mean(style_img[i]), torch.mean(target_img[i]))
        # std_style = mse(torch.std(style_img[i]), torch.std(target_img[i]))
        style_loss += mean_style + std_style
    del target_img

    return content_loss.item(), style_loss.item()

    

def get_score(mode, text_src, text_sty, image, img, coco, catIds, device, image_sty_path=None):
    assert mode[0] in ['clip', 'vgg']
    assert mode[1] in ['pixel', 'vgg']
    image = torchvision.transforms.ToTensor()(Image.open(image))
    image = torch.tensor(image).float().to(device)

    img_process = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize(224, interpolation=3),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    height, width = img['height'], img['width']
    total_mask = torch.zeros((height, width)).bool().to(device)
    for ann in anns:
        rle = maskUtils.frPyObjects(ann['segmentation'], height, width)
        mask = maskUtils.decode(rle)
        mask = torch.tensor(mask).to(device)
        if ann['iscrowd'] == 0:
            total_mask = torch.maximum(total_mask, mask.squeeze())  # union of all masks
    total_mask_resize = torchvision.transforms.Resize((512, 512))(total_mask.unsqueeze(0)).to(device)

    ## original image ##
    mask = total_mask.unsqueeze(0).repeat(3, 1, 1).to(device)
    fore = torch.where(mask, image.clone(), torch.tensor(0, dtype=torch.float32).to(device))
    fore_resize = torchvision.transforms.Resize((512, 512))(fore)
    back = torch.where(mask, torch.tensor(0, dtype=torch.float32).to(device), image.clone()).to(device)
    # back = torch.where(mask, torch.tensor(0, dtype=torch.uint8), torch.tensor(image.copy()))
    back_resize = torchvision.transforms.Resize((512, 512))(back)

    ## stylized image ##
    if mode[2] == 'run_eval':
        image_sty = image_sty_path.squeeze().to(device).float()
        mask = torchvision.transforms.Resize((512, 512))(mask)
    elif mode[2] == 'eval':
        image_sty = torchvision.transforms.ToTensor()(Image.open(image_sty_path))
        image_sty = torch.tensor(image_sty).float().to(device)
        mask = torchvision.transforms.Resize((512, 512))(mask)
    else:
        image_sty = torchvision.transforms.ToTensor()(Image.open(image_sty_path))
        image_sty = torch.tensor(image_sty).float().to(device)
    fore_sty = torch.where(mask, image_sty, torch.tensor(0, dtype=torch.float32).to(device)).to(device)
    fore_sty_resize = torchvision.transforms.Resize((512, 512))(fore_sty)
    back_sty = torch.where(mask, torch.tensor(0, dtype=torch.float32).to(device), image_sty).to(device)
    back_sty_resize = torchvision.transforms.Resize((512, 512))(back_sty)

    ## foreground crop ##
    fore_mask = (total_mask_resize != 0).any(dim=0)
    y_indices, x_indices = torch.where(fore_mask)
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    fore_crop = fore_resize[:, y_min:y_max, x_min:x_max]
    fore_sty_crop = fore_sty_resize[:, y_min:y_max, x_min:x_max]

    ## foreground evaluation ##
    text_tgt = text_sty + " " + text_src
    # text_tgt = text_sty
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.to(device)
    vgg = models.vgg19(pretrained=True).features.eval()
    vgg = vgg.to(device)
    for parameter in vgg.parameters():
        parameter.requires_grad_(False)
    fore_sty_crop_feat = img_process(fore_sty_crop).unsqueeze(0).to(device)
    fore_sty_crop_feat = model.encode_image(fore_sty_crop_feat)
    fore_sty_crop_feat /= fore_sty_crop_feat.norm(dim=-1, keepdim=True)
    if mode[0] == 'clip':
        fore_crop = img_process(fore_crop).unsqueeze(0).to(device)
        fore_crop = model.encode_image(fore_crop)
        fore_crop /= fore_crop.norm(dim=-1, keepdim=True)
        fore_crop_sim = torch.cosine_similarity(fore_crop, fore_sty_crop_feat, dim=-1).item()

        text_tgt = clip.tokenize(text_tgt).to(device)
        text_tgt = model.encode_text(text_tgt).detach()
        text_tgt /= text_tgt.norm(dim=-1, keepdim=True)
        fore_text_sim = torch.cosine_similarity(text_tgt, fore_sty_crop_feat, dim=-1).item()
        # score_fore = 0.5 * fore_crop_sim + 0.5 * fore_text_sim
        score_fore = fore_text_sim

    elif mode[0] == 'vgg':
        score_fore_cnt, score_fore_sty = vgg_loss(fore_crop, fore_sty_crop, vgg, device)

    ## background evaluation ##
    if mode[1] == 'pixel':
        if back_resize.dtype == torch.uint8:
            back_resize = back_resize.float() / 255.0
            back_resize = back_resize.to(torch.float16)
        if back_sty_resize.dtype == torch.uint8:
            back_sty_resize = back_sty_resize.float()
        pixel_diff = F.l1_loss(back_sty_resize, back_resize, reduction='none').to(device)
        pixel_diff = torch.sum(pixel_diff).int()
        num_pixel = torch.sum(fore_mask == 0)
        score_back = 1 - (pixel_diff / num_pixel).item()
    elif mode[1] == 'vgg':
        score_back_cnt, score_back_sty = vgg_loss(back_resize, back_sty_resize, vgg, device)

    if mode[0] == 'clip' and mode[1] == 'pixel':
        return [score_fore, score_back]
    elif mode[0] == 'clip' and mode[1] == 'vgg':
        return [score_fore, score_back_cnt, score_back_sty]
    elif mode[0] == 'vgg' and mode[1] == 'pixel':
        return [score_fore_cnt, score_fore_sty, score_back]
    elif mode[0] == 'vgg' and mode[1] == 'vgg':
        return [score_fore_cnt, score_fore_sty, score_back_cnt, score_back_sty]

if __name__ == '__main__':
    annFile = '{}/annotations/instances_{}.json'.format(args.dataDir, args.dataType)
    className = text_src
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coco = COCO(args.annFile)
    catIds = coco.getCatIds(catNms=[args.className])
    imgIds = coco.getImgIds(catIds=catIds)

    image_idx = random.sample(range(0, len(imgIds)), 10)
    print(image_idx)
    score_fore_list, score_back_list = [], []
    for idx in tqdm(image_idx):
        img = coco.loadImgs(imgIds[image_idx])[0]
        image = plt.imread(f'{dataDir}/{dataType}/{img["file_name"]}')
        score_fore, score_back = get_score(mode, text_src, text_sty, image, coco, device, image_sty_path)
        score_fore_list.append(score_fore)
        score_back_list.append(score_back)
    total_score_fore = np.mean(score_fore_list)
    total_score_back = np.mean(score_back_list)
    print('total score: ', total_score_fore, total_score_back)