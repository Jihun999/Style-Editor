import torch
import math

def angle_between_vectors(vector1, vector2):
    dot_product = torch.dot(vector1, vector2)
    norm1 = torch.norm(vector1)
    norm2 = torch.norm(vector2)
    cos_theta = dot_product / (norm1 * norm2)
    radian = torch.acos(cos_theta)
    degree = radian * (180.0 / math.pi)
    return degree.item()