import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import argparse
import torch
from torchsummary import summary
from thop import profile

from model import *

def compute_speed(model, input_size, iteration=100):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    model.cuda()
    model.eval()

    # with torch.no_grad():
    input = torch.randn(*input_size).cuda()
    for _ in range(50):
        model(input)

    print('=========Eval Forward Time=========')
    torch.cuda.synchronize()
    t_start = time.time()
    for _ in range(iteration):
        model(input)
    torch.cuda.synchronize()
    elapsed_time = time.time() - t_start

    speed_time = elapsed_time / iteration * 1000
    fps = iteration / elapsed_time

    print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, iteration))
    print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
    return speed_time, fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=str, default="512,1024", help="input size of model")
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--classes', type=int, default=19)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--arch', type=str, default='fc')
    parser.add_argument('--layers', type=int, default=161)
    args = parser.parse_args()

    h, w = map(int, args.size.split(','))
    if args.arch == 'psp':
        model = PSPNet(layers=args.layers, classes=args.classes)
    elif args.arch == 'nonlocal':
        model = Nonlocal(layers=args.layers, classes=args.classes)
    elif args.arch == 'sanet':
        model = SANet(layers=args.layers, classes=args.classes)
    elif args.arch == 'bise_v1':
        model = BiseNet(layers=args.layers, classes=args.classes, with_sp=True)
    elif args.arch == 'fanet':
        model = FANet(layers=args.layers, classes=args.classes)
    elif args.arch == 'triple':
        model = TriSeNet(layers=args.layers, classes=args.classes)
    elif args.arch == 'fc':
        if args.layers in [50, 101, 152]:
            model = FC_Net(backbone='resnet', layers=args.layers, classes=args.classes)
        else:
            model = FC_Net(backbone='densenet', layers=args.layers, classes=args.classes)
    compute_speed(model, (args.batch_size, args.num_channels, h, w), iteration=args.iter)
    device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda"
    model.to(device)
    model.eval()
    # summary(model, input_size=(3, 1024, 2048), device=device)
    input = torch.randn(1, 3, 512, 1024).to(device)
    total_ops, total_params  = profile(model, inputs=(input, ), verbose=False)
    print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
    print("%s | %.2f | %.2f" % (args.arch, total_params / (1000 ** 2), total_ops / (1000 ** 3)))
