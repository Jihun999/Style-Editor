import torch

def content_loss(content_features, output_features):
    content_loss += torch.mean((output_features['conv4_2'] - content_features['conv4_2']) ** 2)
    content_loss += torch.mean((output_features['conv5_2'] - content_features['conv5_2']) ** 2)
    return content_loss

def tv_loss(output_image):
    diff1 = output_image[:, :, :, :-1] - output_image[:, :, :, 1:]
    diff2 = output_image[:, :, :-1, :] - output_image[:, :, 1:, :]
    diff3 = output_image[:, :, 1:, :-1] - output_image[:, :, :-1, 1:]
    diff4 = output_image[:, :, :-1, :-1] - output_image[:, :, 1:, 1:]
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    return loss_var_l2

class EmbeddingCLIPLoss(torch.nn.Module):
    def __init__(self, model, source_feature, output_feature, output_patch_feature):
        super().__init__()
        self.model = model
        self.soure_image = source_feature[0]
        self.source_text = source_feature[1]
        self.output_image = output_feature[0]
        self.output_text = output_feature[1]
        self.output_patch_image = output_patch_feature
        self.image_direction = self.get_direction(self.soure_image, self.output_image)
        self.text_direction = self.get_direction(self.source_text, self.output_text) 
        self.patch_direction = self.get_direction(self.soure_image, self.output_patch_image)

    def get_direction(self, start_feature, end_feature):
        return (end_feature - start_feature).detach()
    
    def direction_loss(self, image_direction, text_direction):
        return (1- torch.cosine_similarity(image_direction, text_direction, dim=1)).mean()
    
    def glob_loss(self):
        return self.direction_loss(self.image_direction, self.text_direction)

    def patch_loss(self):
        return self.direction_loss(self.patch_direction, self.text_direction)
    
    def semsty_loss(self):
        return  self.direction_loss(self.output_patch_image, self.output_text)
    
    
    


