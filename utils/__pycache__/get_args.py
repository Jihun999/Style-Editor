
import argparse
import ast

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_path', type=str, default="./face.jpg", help='Image resolution')
    parser.add_argument('--content_name', type=str, default="face", help='Image resolution')
    parser.add_argument('--exp_name', type=str, default="exp1", help='Image resolution')
    parser.add_argument('--text', type=str, default="Fire", help='Image resolution')
    parser.add_argument('--lambda_tv', type=float, default=2e-3, help='total variation loss parameter')
    parser.add_argument('--lambda_patch', type=float, default=9000, help='PatchCLIP loss parameter')
    parser.add_argument('--lambda_dir', type=float, default=500, help='directional loss parameter')
    parser.add_argument('--lambda_c', type=float, default=150, help='content loss parameter')
    parser.add_argument('--crop_size', type=int, default=64, help='cropped image size')
    parser.add_argument('--num_crops', type=int, default=128, help='number of patches')
    parser.add_argument('--img_width', type=int, default=512, help='size of images')
    parser.add_argument('--img_height', type=int, default=512, help='size of images')
    parser.add_argument('--max_step', type=int, default=200, help='Number of domains')
    parser.add_argument('--lr', type=float, default=5e-4, help='Number of domains')
    parser.add_argument('--thresh', type=float, default=0, help='Number of domains')
    parser.add_argument('--loss', type=str, default="ours_l1", help='loss')
    parser.add_argument('--source_patch_sel', type=str, default=True, help='loss')
    parser.add_argument('--return_img', type=str, default=True)
    parser.add_argument('--sv_name', type=str, default='0')
    parser.add_argument('--st_idx', type=str, default='0')

    args = parser.parse_args()
    return args

def parse_common_args(parser):
    parser.add_argument('--content_path', type=str, default="./face.jpg", help='Image resolution')
    parser.add_argument('--content_name', type=str, default="face", help='Image resolution')
    parser.add_argument('--exp_name', type=str, default="exp1", help='Image resolution')
    parser.add_argument('--text', type=str, default="Fire", help='Image resolution')
    parser.add_argument('--lambda_tv', type=float, default=2e-3, help='total variation loss parameter')
    parser.add_argument('--lambda_patch', type=float, default=9000, help='PatchCLIP loss parameter')
    parser.add_argument('--lambda_dir', type=float, default=500, help='directional loss parameter')
    parser.add_argument('--lambda_c', type=float, default=150, help='content loss parameter')
    parser.add_argument('--crop_size', type=int, default=64, help='cropped image size')
    parser.add_argument('--num_crops', type=int, default=128, help='number of patches')
    parser.add_argument('--img_width', type=int, default=512, help='size of images')
    parser.add_argument('--img_height', type=int, default=512, help='size of images')
    parser.add_argument('--max_step', type=int, default=200, help='Number of domains')
    parser.add_argument('--lr', type=float, default=5e-4, help='Number of domains')
    parser.add_argument('--thresh', type=float, default=0.7, help='Number of domains')
    parser.add_argument('--loss', type=str, default="ours_l1", help='loss')
    parser.add_argument('--source_patch_sel', type=str, default=True, help='loss')
    parser.add_argument('--mode', type=str, default='add_attnpatch', help='loss')
    parser.add_argument('--return_img', type=str, default=False)
    parser.add_argument('--interpolation', type=float, default=0)

def get_eval_args():
    parser = argparse.ArgumentParser()
    parse_common_args(parser)
    parser.add_argument('--text_sty', type=str, default="['black sketch']")
    parser.add_argument('--img_idx', type=str, default="[1,32,456,76]")
    parser.add_argument('--eval_mode', type=str, default="['clip', 'pixel', 'run_eval']")
    parser.add_argument('--text_src', type=str, default='dog')
    parser.add_argument('--image_sty_path', type=str, default=None)
    parser.add_argument('--dataDir', type=str, default='/home/jihun/SH/amodal/datasets/COCO2017')
    parser.add_argument('--dataType', type=str, default='train2017')
    args = parser.parse_args()
    args.img_idx = ast.literal_eval(args.img_idx)
    args.text_sty = ast.literal_eval(args.text_sty)
    args.eval_mode = ast.literal_eval(args.eval_mode)
    return args

if __name__ == "__main__":
    args = get_args()