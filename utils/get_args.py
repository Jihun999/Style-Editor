
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', type=str, default="./example_images/beanie_512.png", help='source image path')
    parser.add_argument('--content_name', type=str, default="beanie", help='source text')
    parser.add_argument('--save_dir', type=str, default="model_output", help='save_dir')
    parser.add_argument('--exp_name', type=str, default="test", help='experiment name')
    parser.add_argument('--text', type=str, default="wooden", help='style text')
    parser.add_argument('--lambda_tv', type=float, default=2e-3, help='total variation loss parameter')
    parser.add_argument('--lambda_patch', type=float, default=15000, help='pwd loss parameter')
    parser.add_argument('--lambda_c', type=float, default=400, help='content loss parameter')
    parser.add_argument('--lambda_abp', type=float, default=30000, help='abp loss parameter')
    parser.add_argument('--lambda_con', type=float, default=30000, help='consistency loss parameter')
    parser.add_argument('--img_width', type=int, default=512, help='image width')
    parser.add_argument('--img_height', type=int, default=512, help='image height')
    parser.add_argument('--max_step', type=int, default=200, help='number of iterations')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--return_img', type=str, default=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()