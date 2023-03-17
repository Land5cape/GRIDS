# coding:utf8

import os
from PIL import Image
import torch
from torchvision import transforms
from modules.GRIDS import GRIDS

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

def main():

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    grids = GRIDS(device=device, type='s')
    grids.to(device)

    model_load_path = '.\\model-save\\weight.pth'
    weight = torch.load(model_load_path)
    grids.load_state_dict(weight['net'])

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_path = '.\\images\\woman.JPEG.5.png'
    img = transform(Image.open(img_path)).unsqueeze(0).to(device, non_blocking=True)

    ref_path = '.\\images\\woman.png'
    ref = transform(Image.open(ref_path)).unsqueeze(0).to(device, non_blocking=True)

    # from torchsummary import summary
    # summary(grids, img, ref, depth=3)

    with torch.no_grad():
        output = grids(ref, img)
        print('predict score:', output[0].item())

if __name__ == '__main__':
    main()
