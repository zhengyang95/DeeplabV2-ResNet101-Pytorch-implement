import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import pandas as pd
import numpy as np
from PIL import Image
import multiprocessing
import argparse
import cv2


categories = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']


def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('i', 0, lock=True))
        P.append(multiprocessing.Value('i', 0, lock=True))
        T.append(multiprocessing.Value('i', 0, lock=True))

    def compare(start, step, TP, P, T, input_type, threshold):
        img_num = 0
        for idx in range(start, len(name_list), step):

            name = name_list[idx]

            name = name[-15:-4]
            gt_file = os.path.join(gt_folder, '%s.png' % name)
            gt = np.array(Image.open(gt_file))
            H = np.size(gt, 0)
            W = np.size(gt, 1)

            predict_file = os.path.join(args.predict_dir, '%s.png' % name)
            check_file = predict_file
            ext = os.path.exists(check_file)
            if ext == False:
                # print(check_file[16:])
                continue
            img_num = img_num + 1
            predict = cv2.imread(predict_file, cv2.IMREAD_GRAYSCALE)
            # 以下三行针对01的dark dcrf mask, 只适用于Matlab单目标生成的例子
            # gt_x = gt.copy()
            # gt[gt==255] = 0
            # predict[predict!=0] = np.max(gt_x)

            predict = cv2.resize(predict, (W, H), interpolation= cv2.INTER_NEAREST)
            # predict = np.array(Image.open(predict_file).resize((W, H), 'f', Image.NEAREST))


            'original read gt method'
            # gt_file = os.path.join(gt_folder, '%s.png' % name)
            # gt = np.array(Image.open(gt_file))
            cal = gt < 255
            mask = (predict == gt) * cal

            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict == i) * cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt == i) * cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt == i) * mask)
                TP[i].release()
        print(img_num)

    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i, 8, TP, P, T, input_type, threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    T_TP = []
    P_TP = []
    FP_ALL = []
    FN_ALL = []
    for i in range(num_cls):
        IoU.append(TP[i].value / (T[i].value + P[i].value - TP[i].value + 1e-10))
        T_TP.append(T[i].value / (TP[i].value + 1e-10))
        P_TP.append(P[i].value / (TP[i].value + 1e-10))
        FP_ALL.append((P[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
        FN_ALL.append((T[i].value - TP[i].value) / (T[i].value + P[i].value - TP[i].value + 1e-10))
    loglist = {}
    for i in range(num_cls):
        loglist[categories[i]] = IoU[i] * 100

    miou = np.mean(np.array(IoU))
    loglist['mIoU'] = miou * 100
    if printlog:
        for i in range(num_cls):
            if i % 2 != 1:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100), end='\t')
            else:
                print('%11s:%7.3f%%' % (categories[i], IoU[i] * 100))
        print('\n======================================================')
        print('%11s:%7.3f%%' % ('mIoU', miou * 100))
    return loglist


def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  ' % (key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)


def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath, 'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n' % comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--list",
                        default='/utilisateurs/lyuzheng/DeepL/WSSS/Pytorch/SIPE_HighRes/data/trainaug_voc.txt',
                        type=str)
    parser.add_argument("--predict_dir",
                            default='/media/data/lyuzheng/Result_2024/10/1012_ori_MIP_CAM/mip/SVMResult/dark_mask_0', type=str)
    parser.add_argument("--gt_dir", default='/media/data/lyuzheng/Dataset/Official/voc12/VOCdevkit/VOC2012/SegmentationClassAug/', type=str)
    parser.add_argument('--logfile', default='./evallog.txt', type=str)
    parser.add_argument('--comment', default='0825', type=str)
    parser.add_argument('--type', default='png', choices=['npy', 'png'], type=str)
    parser.add_argument('--t', default=None, type=float)
    parser.add_argument('--curve', default=False, type=bool)
    args = parser.parse_args()

    if args.type == 'npy':
        assert args.t is not None or args.curve
    df = pd.read_csv(args.list, names=['filename'])
    name_list = df['filename'].values
    if not args.curve:
        x = 1
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21, args.type, args.t, printlog=True)
        # writelog(args.logfile, loglist, args.comment)
    else:
        l = []
        for i in range(60):
            t = i / 100.0
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, 21, args.type, t)
            l.append(loglist['mIoU'])
            print('%d/60 background score: %.3f\tmIoU: %.3f%%' % (i, t, loglist['mIoU']))
        writelog(args.logfile, {'mIoU': l}, args.comment)

