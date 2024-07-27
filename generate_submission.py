import argparse
import csv
import torch
import torchvision
from torch.nn import functional as F
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config_for_pred as dataset_config

# options
parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
parser.add_argument('dataset', type=str)
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--full_res', default=False, action="store_true",
                    help='use full resolution 256x256 for test as in Non-local I3D')
parser.add_argument('--test_crops', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--workers', default=8, type=int, metavar='N')
parser.add_argument('--test_list', type=str, default=None)
parser.add_argument('--csv_file', type=str, default='submission.csv')
parser.add_argument('--softmax', default=False, action="store_true")
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--img_feature_dim', type=int, default=256)
parser.add_argument('--pretrain', type=str, default='imagenet')
parser.add_argument('--test_file', type=str, default='test.txt')
args = parser.parse_args()


def parse_shift_option_from_log_name(log_name):
    if 'shift' in log_name:
        strings = log_name.split('_')
        for i, s in enumerate(strings):
            if 'shift' in s:
                break
        return True, int(strings[i].replace('shift', '')), strings[i + 1]
    else:
        return False, None, None


# Load the model
is_shift, shift_div, shift_place = parse_shift_option_from_log_name(args.weights)
if 'RGB' in args.weights:
    args.modality = 'RGB'
elif 'RTD' in args.weights:
    args.modality = 'RTD'
else:
    args.modality = 'Flow'
this_arch = args.weights.split('TSM_')[1].split('_')[2]

num_class, args.train_list, args.val_list, args.root_path, args.root_data_depth, args.root_data_ir, prefix, prefix_ir, prefix_depth = dataset_config.return_dataset(args.dataset, args.modality)
net = TSN(num_class, args.test_segments if is_shift else 1, args.modality,
          base_model=this_arch,
          consensus_type=args.crop_fusion_type,
          img_feature_dim=args.img_feature_dim,
          pretrain=args.pretrain,
          is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
          non_local='_nl' in args.weights)

checkpoint = torch.load(args.weights)
checkpoint = checkpoint['state_dict']
base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                'base_model.classifier.bias': 'new_fc.bias'}
for k, v in replace_dict.items():
    if k in base_dict:
        base_dict[v] = base_dict.pop(k)
net.load_state_dict(base_dict)

input_size = net.scale_size if args.full_res else net.input_size
cropping = torchvision.transforms.Compose([
    GroupScale(net.scale_size),
    GroupCenterCrop(input_size),
])

if args.modality != 'RGBDiff':
    normalize = GroupNormalize(net.input_mean, net.input_std)
else:
    normalize = IdentityTransform()

if args.modality in ['RGB', 'RTD']:
    data_length = 1
elif args.modality in ['Flow', 'RGBDiff']:
    data_length = 5

data_loader = torch.utils.data.DataLoader(
    TSNDataSet(args.root_path, args.root_data_ir, args.root_data_depth, list_file=args.test_file if args.test_file is not None else args.val_list, num_segments=args.test_segments,
               new_length=data_length,
               modality=args.modality,
               image_tmpl=prefix, image_tmpl_ir=prefix_ir, image_tmpl_depth=prefix_depth,
               test_mode=True,
               transform=torchvision.transforms.Compose([
                   cropping,
                   Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                   ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                   normalize,
               ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
    devices = list(range(args.workers))

net = torch.nn.DataParallel(net.cuda())
net.eval()

output = []

def eval_video(video_data, net, this_test_segments):
    with torch.no_grad():
        i, data = video_data
        batch_size = args.batch_size
        num_crop = args.test_crops

        if args.modality == 'RGB':
            sample_length = 3
        elif args.modality == 'RTD':
            sample_length = 9
        elif args.modality == 'Flow':
            sample_length = 10
        else:
            raise ValueError("Unknown modality "+ args.modality)

        data_in = data.view(-1, sample_length, data.size(2), data.size(3))
        if is_shift:
            data_in = data_in.view(batch_size * num_crop, this_test_segments, sample_length, data_in.size(2), data_in.size(3))
        rst = net(data_in)
        rst = rst.reshape(batch_size, num_crop, -1).mean(1)

        if args.softmax:
            rst = F.softmax(rst, dim=1)

        return i, rst.data.cpu().numpy().copy()

for i, data in enumerate(data_loader):
    rst = eval_video((i+1, data), net, args.test_segments, args.modality)
    output.append([rst[1]])

video_pred = [np.argmax(x[0], axis=1)[0] for x in output]
video_labels = [x[1].item() for x in output]

with open(args.csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Video', 'Prediction'])
    for vid_name, pred in zip(video_labels, video_pred):
        csvwriter.writerow([vid_name, pred])

print(f'Predictions saved to {args.csv_file}')
