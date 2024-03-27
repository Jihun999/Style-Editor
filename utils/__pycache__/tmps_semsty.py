import torch
import torch.nn.functional as F
from utils import utils
from torchvision.transforms.functional import crop
from torchvision import transforms
import random
import cv2
import os
import numpy as np
import torchvision.models.detection.anchor_utils as anchor_utils

class TMPS:
    def __init__(self, clip_model, device, num_crops, augment, text_source, args):
        self.clip_model = clip_model
        self.device = device
        self.num_crops = num_crops
        self.augment = augment
        self.img_proc_origin = []
        self.text_source = text_source
        self.args = args
        self.anchors_per_grid = 3
        self.seleted_grids = torch.tensor([]).to(self.device)
        self.last_selected_patches = None  # To store the positions of the patches selected in the last iteration

    def calculate_covered_area(self, patches, max_height, max_width):
        """
        주어진 패치들에 의해 커버되는 영역을 계산합니다.
        """
        if patches is None:
            return torch.zeros((max_height, max_width), dtype=torch.uint8)

        covered_area = torch.zeros((max_height, max_width), dtype=torch.uint8)
        for box in patches:
            x, y, h, w = box
            covered_area[y:y+h, x:x+w] = 1
        return covered_area

    def calculate_max_dimensions(self, patches1, patches2):
        max_width = max([x + w for patches in [patches1, patches2] for x, _, _, w in patches], default=0)
        max_height = max([y + h for patches in [patches1, patches2] for _, y, h, _ in patches], default=0)
        return max_height, max_width

    def visualize_patch_difference(self, content_image, last_patches, current_patches, epoch):
        max_height, max_width = self.calculate_max_dimensions(last_patches, current_patches)

        last_covered_area = self.calculate_covered_area(last_patches, max_height, max_width)
        current_covered_area = self.calculate_covered_area(current_patches, max_height, max_width)

        difference_area = last_covered_area - current_covered_area
        difference_indices = torch.where(difference_area > 0)

        img = torch.ones_like(content_image)
        img[:,:,difference_indices[0], difference_indices[1]] = 0.2

        current_indices = torch.where(current_covered_area > 0)
        img[:,:,current_indices[0], current_indices[1]] = 0

        return img    
    def vis_girds(self, content_image, grid_counts, grid_size=9, strides=64):
        content_img = content_image.squeeze().cpu().detach().numpy()
        content_img = np.transpose(content_img, (1, 2, 0))
        content_img = (content_img * 255).astype(np.uint8).copy()
        top_k_idx = torch.where(grid_counts >= 1)[0]
        top_2_idx = torch.where(grid_counts >= 2)[0]
        for idx in top_k_idx:
            center_y = idx.item() // grid_size * strides
            center_x = (idx.item() % grid_size) * strides
            cv2.circle(content_img, (center_x, center_y), 5, (0, 255, 0), -1)
        for idx2 in top_2_idx:
            center_y = idx2.item() // grid_size * strides
            center_x = (idx2.item() % grid_size) * strides
            cv2.circle(content_img, (center_x, center_y), 5, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(self.args.save_dir, 'grid_counts{}.jpg'.format(1)), content_img)

    def TMPS_anchor_process(self, content_image, target, epoch):
        output = {}
        source_img_proc = []
        pos_list = []
        img_proc = []
        content_image = content_image.detach()
        if epoch <= 20:
            anchors = generate_anchors(content_image, epoch)
            pos_tensor = anchors.to(self.device).int()
            for pos in pos_tensor:
                x, y, h, w = pos.tolist()  # Convert tensor to list
                source_crop = crop(content_image, y, x, h, w)
                source_crop_ = self.augment(source_crop)
                source_img_proc.append(source_crop_)
                del source_crop, source_crop_
            source_img_proc = torch.cat(source_img_proc, dim=0)
        else:
            if epoch == 21:
                grid_counts = torch.bincount(self.seleted_grids.int())
                self.top_k_indices = torch.where(grid_counts >= 1)[0]
                self.num_crops = int(self.top_k_indices.size(0)*2)
                # self.vis_girds(content_image, grid_counts)

            for num in range(self.num_crops):
                source_crop, pos = select_anchor_patches(content_image, self.top_k_indices, epoch, num=num)
                source_crop_ = self.augment(source_crop)
                source_img_proc.append(source_crop_)
                pos_list.append(pos)
            source_img_proc = torch.cat(source_img_proc, dim=0)
            pos_tensor = torch.tensor(pos_list).to(self.device)
        source_features_ = self.clip_model.encode_image(utils.clip_normalize(source_img_proc, self.device))
        source_features_ /= (source_features_.clone().norm(dim=-1, keepdim=True))
        source_cos_sim = F.cosine_similarity(source_features_, self.text_source.repeat(source_features_.size(0), 1), dim=1)
        source_top_indice_ = source_cos_sim.topk(3).indices
        source_top_image_features = source_features_[source_top_indice_]
        source_top_image_features = source_top_image_features.mean(axis=0, keepdim=True)
        source_cos_sim = F.cosine_similarity(source_features_, source_top_image_features, dim=1)
        source_top_indice = source_cos_sim.topk(self.num_crops//2).indices
        if epoch <= 20:
            source_top_indice = source_top_indice[source_cos_sim[source_top_indice] > 0.9]
            selected_source_features = source_features_[source_top_indice]
            text_source_cos_sim = F.cosine_similarity(selected_source_features, self.text_source.repeat(selected_source_features.size(0), 1), dim=1)
            mean_cos_sim = text_source_cos_sim.mean()
            final_source_top_indice = (text_source_cos_sim > mean_cos_sim).nonzero().squeeze()
            source_top_indice = source_top_indice[final_source_top_indice]
        pos_tensor_50 = pos_tensor[source_top_indice]
        if epoch <= 20:
            seleted_grids = torch.div(source_top_indice, self.anchors_per_grid, rounding_mode='trunc')
            self.seleted_grids = torch.cat((self.seleted_grids, seleted_grids))
            if self.args.loss == 'ours_l1':
                target_loss_img = target.clone().requires_grad_(True).to(self.device)
            # target_loss_img = torch.ones_like(target)
        # print(pos_tensor_50.shape, 'pos_tensor_50.shape')
        if epoch > 20:  # Skip for the first epoch
            target_loss_img = self.visualize_patch_difference(content_image, self.last_selected_patches, pos_tensor_50, epoch)
            # self.visualize_common_area(content_image, self.last_selected_patches, pos_tensor_50, epoch)

        # Update last_selected_patches for the next epoch
        self.last_selected_patches = pos_tensor_50.clone().detach()
        if len(pos_tensor_50.shape) == 1 and pos_tensor_50.shape[0] == 4:
            pos_tensor_50 = pos_tensor_50.unsqueeze(0)
        for n in range(pos_tensor_50.size(0)):
            pos = pos_tensor_50[n]
            if pos.shape[0] == 3:
                x, y, w = pos[0], pos[1], pos[2]
                h = w
            elif pos.shape[0] == 4:
                x, y, h, w = pos[0], pos[1], pos[2], pos[3]
            target_crop = crop(target, y, x, h, w)
            target_crop_ = self.augment(target_crop)
            img_proc.append(target_crop_)
            if self.args.loss == 'ours_l1' and epoch <= 20:
                patch__ = content_image[:,:,y:y+h, x:x+w]
                target_loss_img[:,:,y:y+h, x:x+w] = patch__
                # target_loss_img[:,:,y:y+h, x:x+w] = 0
        img_proc = torch.cat(img_proc, dim=0)
        image_features = self.clip_model.encode_image(utils.clip_normalize(img_proc, self.device))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
        if self.args.loss == 'ours_l1':
            # img_loss_np = target.squeeze().cpu().detach().numpy()
            # img_loss_np = np.transpose(img_loss_np, (1, 2, 0))
            # img_loss_np = (img_loss_np * 255).astype(np.uint8).copy()
            # img_loss_np = cv2.cvtColor(img_loss_np, cv2.COLOR_RGB2BGR)
            # # cv2.imwrite(os.path.join(self.args.save_dir, 'target_loss_tmps.jpg'), img_loss_np)
            output['target_loss_img'] = target_loss_img
        if self.args.source_patch_sel:
            output['source_patch_features'] = source_features_[source_top_indice]
        output['image_features'] = image_features
        

        # cv2.rectangle(img_tr, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)  # Green color, 2px thickness
        # cv2.rectangle(img_src, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)  # Green color, 2px thickness
        # patch__ = content_image[:,:,y:y+h, x:x+w]
        # patch_np = (patch__.squeeze().cpu().detach().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        # img_wt[y:y+h, x:x+w,:] = patch_np
        # img_src_mask[y:y+h, x:x+w, :] = 0
        # img_tr_mask[y:y+h, x:x+w, :] = 0
        if epoch % 100 == 0:
            tr = target_loss_img.squeeze().cpu().detach().numpy()
            tr = np.transpose(tr, (1, 2, 0))
            tr = (tr * 255).astype(np.uint8).copy()
            # for box in pos_tensor_50:
            #     x, y, h, w = box
            #     cv2.rectangle(tr, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)  # Green color, 2px thickness
            cv2.imwrite(os.path.join(self.args.save_dir, 'text_img_patch{}.jpg'.format(epoch)), tr)
        #     cv2.imwrite(os.path.join(self.args.save_dir, 'text_img_patch{}.jpg'.format(epoch)), img_np)
            # cv2.imwrite(os.path.join(self.args.save_dir, 'target_img_patch{}.jpg'.format(epoch)), img_tr)
            # cv2.imwrite(os.path.join(self.args.save_dir, 'target_loss_tmps{}.jpg'.format(epoch)), img_loss_np)
        return output

class SuperTMPS:
    def __init__(self, clip_model, device, num_crops, augment, text_source, args, content_image):
        self.clip_model = clip_model
        self.device = device
        self.num_crops = num_crops
        self.augment = augment
        self.img_proc_origin = []
        self.text_source = text_source
        self.args = args
        self.content_image = content_image.squeeze(0)
        self.init_process()

    def init_process(self):
        content_image_np = self.content_image.permute(1, 2, 0).cpu().numpy()
        superpixel = cv2.ximgproc.createSuperpixelSLIC(content_image_np, region_size=50, ruler=20.0)
        num_superpixels = 40  # You can adjust this number
        superpixel.iterate(num_superpixels)
        self.super_labels = torch.tensor(superpixel.getLabels()).cuda()

    def TMPS_process(self, content_image, target, epoch):
        output = {}
        unique_labels = torch.unique(self.super_labels)
        source_img_proc, pos_list, img_proc = [], [], []
        import matplotlib.pyplot as plt

        for label in unique_labels:
            superpixel_mask = (self.super_labels == label)
            where = torch.where(superpixel_mask)
            y_min, y_max = torch.min(where[0]), torch.max(where[0])
            x_min, x_max = torch.min(where[1]), torch.max(where[1])
            patch = self.content_image[:, y_min:y_max+1, x_min:x_max+1]
            patch_mask = superpixel_mask[y_min:y_max+1, x_min:x_max+1].unsqueeze(0)
            patch_only_superpixel = torch.where(patch_mask, patch, torch.zeros_like(patch))
            pos_list.append([x_min, y_min, x_max-x_min+1, y_max-y_min+1])
            source_crop_ = self.augment(patch_only_superpixel.unsqueeze(0))
            # plt.imsave('test.png', source_crop_.squeeze(0).permute(1,2,0).cpu().numpy())
            source_img_proc.append(source_crop_)
        source_img_proc = torch.cat(source_img_proc, dim=0)
        pos_tensor = torch.tensor(pos_list).to(self.device)
        source_features_ = self.clip_model.encode_image(utils.clip_normalize(source_img_proc, self.device))
        source_features_ /= (source_features_.clone().norm(dim=-1, keepdim=True))
        source_cos_sim = F.cosine_similarity(source_features_, self.text_source.repeat(source_features_.size(0), 1), dim=1)
        source_top_indice_ = source_cos_sim.topk(5).indices
        source_top_image_features = source_features_[source_top_indice_]
        source_top_image_features = source_top_image_features.mean(axis=0, keepdim=True)
        source_cos_sim = F.cosine_similarity(source_features_, source_top_image_features, dim=1)
        source_top_indice = source_cos_sim.topk(self.num_crops//2).indices
        source_top_indice = source_top_indice[source_cos_sim[source_top_indice] > 0.9]
        source_top_indice = source_top_indice.topk(source_top_indice.size(0)//2).indices
        if self.args.loss == 'ours_l1':
            target_loss_img = target.clone().requires_grad_(True).to(self.device)
            for label in source_top_indice:
                superpixel_mask = (self.super_labels == label)
                where_terget = torch.where(superpixel_mask)
                y_min, y_max = torch.min(where_terget[0]), torch.max(where_terget[0])
                x_min, x_max = torch.min(where_terget[1]), torch.max(where_terget[1])
                patch = target.squeeze(0)[:, y_min:y_max+1, x_min:x_max+1]
                patch_mask = superpixel_mask[y_min:y_max+1, x_min:x_max+1].unsqueeze(0)
                target_superpixel = torch.where(patch_mask, patch, torch.zeros_like(patch))
                target_superpixel_ = self.augment(target_superpixel.unsqueeze(0))
                img_proc.append(target_superpixel_)
                if self.args.loss == 'ours_l1':
                    patch__ = content_image[:, :, superpixel_mask].to(target_loss_img.dtype)
                    target_loss_img[:, :, superpixel_mask] = patch__
            img_proc = torch.cat(img_proc, dim=0)
            img_aug = img_proc
            image_features = self.clip_model.encode_image(utils.clip_normalize(img_aug, self.device))
            image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
            if self.args.loss == 'ours_l1':
                img_loss_np = target_loss_img.squeeze().cpu().detach().numpy()
                img_loss_np = np.transpose(img_loss_np, (1, 2, 0))
                img_loss_np = (img_loss_np * 255).astype(np.uint8).copy()
                img_loss_np = cv2.cvtColor(img_loss_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self.args.save_dir, 'target_loss_tmps.jpg'), img_loss_np)
                output['target_loss_img'] = target_loss_img
            if self.args.source_patch_sel:
                output['source_patch_features'] = source_features_[source_top_indice]
            output['image_features'] = image_features

        else:
            for n in range(self.num_crops):
                target_crop = rnd_crop(target)
                target_crop_ = self.augment(target_crop)
                img_proc.append(target_crop_)
            img_proc = torch.cat(img_proc,dim=0)
            img_aug = img_proc
            image_features = self.clip_model.encode_image(utils.clip_normalize(img_aug, self.device))
            image_features /= (image_features.clone().norm(dim=-1, keepdim=True))
            output['image_features'] = image_features
        return output

def select_anchor_patches(image, top_k_indices, epoch, strides=64, grid_size=9, num=0):
    ratio = 1 - 0.2 * min(max((epoch - 21) / (200 - 21), 0), 1)

    i = top_k_indices[num % top_k_indices.size(0)]
    grid_center_y = i.item() // grid_size * strides
    grid_center_x = (i.item() % grid_size) * strides

    # Select a random center within the grid cell
    center_y = random.randint(max(0, grid_center_y - strides//2), min(image.shape[2], grid_center_y + strides//2))
    center_x = random.randint(max(0, grid_center_x - strides//2), min(image.shape[3], grid_center_x + strides//2))

    # Calculate the maximum possible crop size based on the grid size
    crop_height = random.randint(int(64*ratio), int(128*ratio))
    crop_width = random.randint(int(64*ratio), int(128*ratio))

    # Calculate the top-left corner of the patch
    y = max(center_y - crop_height // 2, 0)
    x = max(center_x - crop_width // 2, 0)

    # Make sure the patch does not exceed the image boundaries
    crop_height = min(crop_height, image.shape[2] - y)
    crop_width = min(crop_width, image.shape[3] - x)
    if (x < 0) or (y < 0) or (x+crop_width > image.shape[3]) or (y+crop_height > image.shape[2]):
        print('error')

    patch = crop(image, y, x, crop_height, crop_width)
    return patch, [x, y, crop_height, crop_width]
    
def rnd_crop(image):        
    size = random.randint(64, 224)
    cropper = transforms.Compose([
        transforms.RandomCrop(size)
    ])
    image = cropper(image)
    return image

def random_square_crop(image, return_coords=False):        
    w, h = image.shape[2], image.shape[3]
    crop_size = random.randint(64, 128)  # Random crop size
    crop_sizeh = random.randint(64, 128)  # Random crop size

    if crop_size > min(w, h):  # If crop size is greater than the smaller dimension of the image
        crop_size = min(w, h)  # Make the crop size equal to the smaller dimension
    if crop_sizeh > min(w, h):
        crop_sizeh = min(w, h)

    i = torch.randint(0, h - crop_size + 1, size=(1,)).item()  
    j = torch.randint(0, w - crop_sizeh + 1, size=(1,)).item()
    cropped_image = crop(image, i, j, crop_size, crop_sizeh)

    if return_coords:
        return cropped_image, [i, j, crop_size, crop_sizeh]
    else:
        return cropped_image


def generate_anchors(image, epoch):
    ratio = 1 - 0.4 * min(max((epoch - 21) / (200 - 21), 0), 1)
    # ratio = 1
    grid_sizes = [(9,9)]
    anchor_sizes = ((128 * ratio),)
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    strides = [(64,64)]
    transform = transforms.Compose([transforms.ToTensor()])

    anchor_generator = anchor_utils.AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
    anchors = anchor_generator.grid_anchors(grid_sizes=grid_sizes, strides=strides)
    anchors = adjust_anchors(anchors[0], list(image.shape[-2:]))
    del anchor_generator
    return torch.tensor(anchors)

def adjust_anchors(anchors, image_shape):
    adjusted_anchors = []
    for anchor in anchors:
        x1, y1, x2, y2 = anchor
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image_shape[1] - 1, x2)
        y2 = min(image_shape[0] - 1, y2)
        adjusted_anchors.append([int(x1), int(y1), int(y2-y1), int(x2-x1)])
    del anchors
    return adjusted_anchors
