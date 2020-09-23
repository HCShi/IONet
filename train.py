from util import get_args
from network import *
from datasets import VideoDataset, TrainDataset
from torchvision import transforms, models
import torch.backends.cudnn as cudnn

def main():
    args = get_args()
    torch.backends.cudnn.enabled = False
    cudnn.benchmark = False
    torch.multiprocessing.set_sharing_strategy('file_system')

    train_loader = torch.utils.data.DataLoader(
        TrainDataset(args=args, transform=transforms.Compose([
                         transforms.CenterCrop((224, 224)),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True, num_workers=0)

    video_val_loader = torch.utils.data.DataLoader(
        VideoDataset(args=args, transform=transforms.Compose([
                         transforms.CenterCrop((224,224)),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor()]), test_mode=True),
        batch_size=args.batch_size, shuffle=False, num_workers=0)

    print("start training")
    for epoch in range(args.epochs):
        train(train_loader, video_val_loader, args)


def train(train_loader, video_val_loader, args):
    model = GANModel(args).cuda()
    best_prec = 0

    for i1, (image, image_label, video_feature, video_label) in enumerate(train_loader):
        with torch.no_grad():
            input = {'video': video_feature, 'trimmed': image, 'trimmed_label': image_label, 'video_label': video_label}

        model.train_set_input(input)
        model.optimize_parameters()

        if (i1 % 5) == 0:
            t = 0
            with torch.no_grad():
                print('*****************')
                for i2, (video, video_label, video_name) in enumerate(video_val_loader):
                    test_input = {'video': video, 'video_label': video_label}
                    model.test_set_input(test_input)
                    Attention, video_class_result, predict_label, real_label = model.test_forward()

                    if real_label.dim() == 2:  #
                        real1 = real_label[0][0]
                        real2 = real_label[0][1]
                        id1 = torch.equal(real1, predict_label[0])
                        id2 = torch.equal(real2, predict_label[0])
                        if id1 or id2:
                            t += 1
                    else:
                        id = torch.equal(real_label, predict_label)
                        if id:
                            t += 1
                    print(predict_label)
                print(t)
                if t > best_prec:
                    torch.save(model.state_dict(), 'models/' + str(t) + ".pth")
                    best_prec = t

if __name__ == '__main__':
    # parse the arguments
    args = get_args()
    main()
