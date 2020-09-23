from util import get_args, detect_with_thresholding, mask_to_detections
from network import *
from util import *
from datasets import VideoDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
def nms(proposals, thresh):
    proposals = np.array(proposals)
    x1 = proposals[:,1]
    x2 = proposals[:,2]
    scores = proposals[:,3]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]
    return keep

def smooth(x):
    temp = np.array(x)
    temp[1:, :] = temp[1:, :] + x[:-1, :]
    temp[:-1, :] = temp[:-1, :] + x[1:, :]
    temp[1:-1, :] /= 3
    temp[0, :] /= 2
    temp[-1, :] /= 2
    return temp

def main():
    best_pec1 = 0
    args = get_args()

    torch.backends.cudnn.enabled = False
    cudnn.benchmark = False
    torch.multiprocessing.set_sharing_strategy('file_system')

    video_val_loader = torch.utils.data.DataLoader(
        VideoDataset(args=args, transform=transforms.Compose([
                         transforms.CenterCrop((224,224)),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor()]), test_mode=True),
        batch_size=args.batch_size, shuffle= True, num_workers=8)

    print("start validate")
    validate(video_val_loader, args)

def validate(video_val_loader, args):
    model = GANModel(args).cuda()
    model.load_state_dict(torch.load('models/best.pth'))
    thrh = args.thrh
    pro = args.pro
    weight_global = args.weight_global
    sample_offset = args.sample_offset
    fps = args.fps

    pred_file = 'thumos-I3D-pred.txt' # for thumos dataset
    anno_dir = 'thumos14-test-annotations'

    with torch.no_grad():
        for i, (video, video_label, video_cnt, video_name) in enumerate(video_val_loader):
            test_input = {'video': video, 'video_label': video_label}
            model.test_set_input(test_input)
            Attention, video_middle_class_result, video_class_result, predict_label, real_label = model.test_forward()
            softmax = nn.Softmax(dim=-1)
            video_cnt = video_cnt.item()
            duration = video_cnt/fps
            video_name = video_name[0]
            video_middle_class_result = softmax(torch.squeeze(video_middle_class_result, 0)).cpu()
            Attention = torch.squeeze(Attention).cpu()
            #print(video_class_result)
            video_class_result = softmax(torch.squeeze(video_class_result)).cpu()

            #####detection#####
            out_detections = []
            for class_id in range(args.class_num):
                if video_class_result[class_id] <= args.global_score_thrh:  # threshold for 0.1
                #if class_id != predict_label:
                    continue
                _score = video_middle_class_result[:, class_id]  # 0.3664
                metric = Attention * _score
                # print(torch.gt(metric, (_score/Attention.size(-1))).nonzero())
                metric = smooth(metric)
                metric = normalize(metric).detach().numpy()

                # att_filtering_value = 1 / Attention.shape[0]
                # assert (att_filtering_value is not None)
                #
                # metric = video_class_result[class_id]
                # metric = smooth(metric)
                # metric = normalize(metric)
                # metric[Attention < att_filtering_value] = 0
                # metric = normalize(metric).detach().numpy()

                # map the feature to the original video frame
                t_cam = interpolate(metric, frame_cnt=video_cnt, sample_rate=16, snippet_size=16, kind='linear')
                t_cam = np.expand_dims(t_cam, axis=1)

                mask = detect_with_thresholding(t_cam, thrh, pro)  # mask calculation

                temp_out = mask_to_detections(mask, t_cam)  # [start, end, None, detection_score]#
                for entry in temp_out:
                    entry[2] = class_id

                    entry[3] += video_class_result[class_id].item() * weight_global  # each class confidence

                    entry[0] = (entry[0] + sample_offset) / fps
                    entry[1] = (entry[1] + sample_offset) / fps

                    entry[0] = max(0, entry[0])
                    entry[1] = max(0, entry[1])
                    entry[0] = min(duration, entry[0])
                    entry[1] = min(duration, entry[1])

                #########################################
                for entry_id in range(len(temp_out)):
                    temp_out[entry_id].insert(0, video_name)
                    temp_out = nms(temp_out, 0.7)
                print(temp_out)
                out_detections += temp_out  #to obtain the different category detections of videos

        output_detections_thumos14(out_detections, pred_file)

        summary_file = 'final_localization.npz'
        all_test_map = np.zeros((9, 1))
        all_test_aps = np.zeros((9, args.class_num))
        for IoU_idx, IoU in enumerate([.1, .2, .3, .4, .5, .6, .7, .8, .9]):
            if len(out_detections) != 0:
                temp_aps, temp_map = eval_thumos_detect(pred_file,anno_dir,'test',IoU)
                all_test_aps[IoU_idx, :] = temp_aps
                all_test_map[IoU_idx, 0] = temp_map
        print('{}'.format(IoU_idx))
        np.savez(summary_file, all_test_aps=all_test_aps, all_test_map=all_test_map)

if __name__ == '__main__':
    # parse the arguments
    args = get_args()
    main()


