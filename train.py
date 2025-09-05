import os
import torch
import torch.nn
import torch.optim as optim
import models.StyleNet as StyleNet
from utils import utils
import clip
import torch.nn.functional as F
from utils.template import imagenet_templates
from torchvision import utils as vutils
from tqdm import tqdm
from utils import tmps
from utils.get_args import get_args
from torch.cuda.amp import autocast
from utils.extract_noun_phrase import extract_noun_phrase
from pytorch_msssim import ms_ssim
from torch import nn
from torchvision.models import vgg19, VGG19_Weights
from torchvision import transforms

def train(args):

    save_dir = args.save_dir + '/'
    if os.path.exists(save_dir) == False:
        os.mkdir(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert (args.img_width%8)==0, "width must be multiple of 8"
    assert (args.img_height%8)==0, "height must be multiple of 8"

    VGG = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
    VGG.to(device)

    for parameter in VGG.parameters():
        parameter.requires_grad_(False)


    content_path = args.content_path
    content_image = utils.load_image2(content_path, 
                                      img_height=args.img_height, 
                                      img_width=args.img_width)
    content = args.content_name
    exp = args.exp_name

    content_image = content_image.to(device) # source_img
    content_features = utils.get_features(utils.img_normalize(content_image, device), VGG)

    target = content_image.clone().requires_grad_(True).to(device)
    clip_model, _ = clip.load('ViT-B/32', device, jit=False)
    clip_model.eval()

    style_net = StyleNet.UNet()
    style_net.to(device)

    optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    steps = args.max_step

    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize((224, 224))
    ])

    source = content
    source_noun = extract_noun_phrase(source)
    prompt = args.text + " " + source_noun

    with torch.no_grad():
        template_source = utils.compose_text_with_templates(source, imagenet_templates) # source_text
        tokens_source = clip.tokenize(template_source).to(device)
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)

        source_features = clip_model.encode_image(utils.clip_normalize(content_image,device))
        source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

        template_tgt = utils.compose_text_with_templates(prompt, imagenet_templates)
        tokens_tgt = clip.tokenize(template_tgt).to(device)
        text_target = clip_model.encode_text(tokens_tgt).detach()
        text_target = text_target.mean(axis=0, keepdim=True)
        text_target /= text_target.norm(dim=-1, keepdim=True)

    print('source: ', source, ', prompt: ', prompt)
    processor = tmps.TMPS(clip_model, device, augment, text_source, args=args)
    for epoch in tqdm(range(0, steps+1)):
        with autocast():
            target = style_net(content_image, use_sigmoid=True).to(device) # target_img
        target.requires_grad_(True)

        target_features = utils.get_features(utils.img_normalize(target, device), VGG)
        content_loss = 0
        content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

        loss_patch=0
        output = processor.TMPS_anchor_process(content_image, target, epoch)
        image_features = output['image_features']
        # Text matched patch selection
        source_patch_features = output['source_patch_features']
        target_loss_img = output['target_loss_img']

        img_direction = (image_features-source_patch_features)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        text_direction = (text_target-text_source).repeat(image_features.size(0),1) # text_dir
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        loss_temp = (1- torch.cosine_similarity(img_direction, text_direction, dim=1))
        loss_patch+=loss_temp.mean()
        del loss_temp

        glob_features = clip_model.encode_image(utils.clip_normalize(target,device))
        glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))

        reg_tv = args.lambda_tv*utils.get_image_prior_losses(target)

        source_cos_sim = torch.cosine_similarity(source_features, torch.cat((source_patch_features, source_features), dim=0))
        target_cos_sim = torch.cosine_similarity(glob_features, torch.cat([image_features, glob_features], dim=0))
        temperature_src = 0.5
        temperature_tgt = 2
        source_cos_sim = nn.Softmax(dim=0)(source_cos_sim/temperature_src).detach()
        target_cos_sim = nn.Softmax(dim=0)(target_cos_sim/temperature_tgt)
 
        loss_jsd = get_jsd(source_cos_sim, target_cos_sim)
        
        with autocast():
            ms_ssim_loss = 1 - ms_ssim(target, content_image, data_range=255, size_average=True)
        if epoch <= 20:
            img_l1_loss = F.l1_loss(target_loss_img, content_image)
        else:
            mae_loss = F.l1_loss(target, content_image, reduction='none')
            img_l1_loss = torch.mean(mae_loss * target_loss_img)
        abp_loss = img_l1_loss + ms_ssim_loss

        total_loss = args.lambda_patch * loss_patch + reg_tv + args.lambda_c * content_loss + args.lambda_abp * abp_loss + args.lambda_con * loss_jsd 
        optimizer.zero_grad()
        total_loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()


    if args.return_img:
        out_path = os.path.join(save_dir, prompt + '_' + content + '_' + exp + '.png')
        output_image = target.clone()
        vutils.save_image(output_image, out_path, nrow=1, normalize=True)

def get_jsd(p1, p2):
    m = 0.5 * (p1 + p2)
    out = 0.5 * (nn.KLDivLoss(reduction='sum', log_target=True)(torch.log(m), torch.log(p1))
                    + nn.KLDivLoss(reduction='sum', log_target=True)(torch.log(m), torch.log(p2)))
    return out


if __name__ == '__main__':
    args = get_args()
    train(args)