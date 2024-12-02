import time, argparse, sys, os, random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from datasets.dataset_pairs_wRandomSample import DatasetForInference
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.image as img
from utils.UTILS import compute_psnr, compute_ssim

# Set random seed for reproducibility
sys.path.append(os.getcwd())
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(20)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

# Argument settings
parser.add_argument('--img_dir', type=str, required=True, help='Path to evaluation input images')
parser.add_argument('--model_path', type=str, required=True, help='Path to the model')

parser.add_argument('--save_path', type=str, required=True, help='Path to save evaluation results')
parser.add_argument('--base_channel', type=int, default=18, help='Base channels for the network')
parser.add_argument('--num_block', type=int, default=6, help='Number of residual blocks')
parser.add_argument('--flag', type=str, default='S1', help='Model flag')
args = parser.parse_args()

# Ensure save path exists
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

# Function to load evaluation data
def get_data(input_dir):
    data = DatasetForInference(input_dir)
    data_loader = DataLoader(dataset=data, batch_size=1, num_workers=4)
    return data_loader

# Evaluation function
def test(net, data_loader, save_path):
    net.to(device)
    net.eval()

    with torch.no_grad():
        cnt = 0
        for index, data in enumerate(tqdm(data_loader), 0):
            data_in, name = data

            inputs = Variable(data_in).to(device)
            outputs = net(inputs)

            # Save the result images
            out_np = np.squeeze(torch.clamp(outputs, 0., 1.).cpu().detach().numpy()).transpose((1, 2, 0))
            img.imsave(os.path.join(save_path, name[0]), np.uint8(out_np * 255.))
            cnt += 1
        print(f"Sucessfully processed {cnt:04d} images. \nCheck {save_path} for results.")


if __name__ == '__main__':
    # Load the model
    if args.flag == 'K1':
        from networks.Network_Stage2_K1_Flag import UNet
    elif args.flag == 'K3':
        from networks.Network_Stage2_K3_Flag import UNet
    elif args.flag == 'S1':
        from networks.Network_Stage1 import UNet

    net = UNet(base_channel=args.base_channel, num_res=args.num_block)

    try:
        pretrained_model = torch.load(args.model_path)
        net.load_state_dict(pretrained_model, strict=True)
    except Exception:
        print("------Model loading failed------") 
    else:
        print('------Model successfully loaded------')

    # Prepare the evaluation data
    trans_eval = transforms.Compose([transforms.ToTensor()])
    data_loader = get_data(input_dir=args.img_dir)

    # Perform evaluation
    test(net=net, data_loader=data_loader, save_path=args.save_path)
