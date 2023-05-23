import torch
import argparse, os
import numpy as np
import random
import math
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from lr_scheduler import PolyScheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2 as ToTensor
from torchvision import models
from torchvision.models import ResNet18_Weights

from models.CDCNs import Conv2d_cd, CDCNpp
from models.AdMSLoss import AdMSoftmaxLoss

from datasets.oulup_dataset import AMOuluDataset
from datasets.lcc_fasd import LccFasdDataset

import torch.nn.functional as F
import torch.optim as optim
import copy
import torch.nn as nn

from utils import performances

def setup_seed(seed, cuda_deterministic=True):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


lcc_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),])

def transform(mode):
    if mode == 'train':
        transform = A.Compose(
                    [
                    A.RandomCrop(height=160, width=160, p=1.0),
                    A.HorizontalFlip(p=0.5),
                    # A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    # A.augmentations.transforms.ISONoise(color_shift=(0.15,0.35),
                                                        # intensity=(0.2, 0.5), p=0.5), #p=0.2),
                    # A.augmentations.transforms.RandomBrightnessContrast(brightness_limit=0.2,
                                                                        # contrast_limit=0.2,
                                                                        # brightness_by_max=True,
                                                                        # always_apply=False, p=0.5), #p=0.3),
                    # A.MotionBlur(blur_limit=5, p=0.5), #p=0.2),
                    # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                    # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.0, rotate_limit=45, p=0.5),
                    A.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensor(),
                    ]
                )
    else:
        transform = A.Compose(
                    [
                    A.Resize(height=160, width=160, p=1.0),
                    A.transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                    ToTensor(),
                    ]
                )
    return transform

def train_test():
    writer = SummaryWriter(log_dir='./runs')
    setup_seed(2048, cuda_deterministic=False)
    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log+'/'+ args.log+'_log_P1.txt', 'w')
    
    print("Oulu-NPU, P1:\n")

    log_file.write('Oulu-NPU, PatchNet:\n')
    log_file.flush()
    # GPU  & log file  -->   if use DataParallel, please comment this command
    # os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (args.gpu)
    device = [0, 1]
    # dataset
    train_data = AMOuluDataset(root_folder='/mnt/training_dataset/face_dataset/Oulu_align',\
                            mode='train',\
                            transform=transform(mode='train'),)
    train_loader = DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=4, pin_memory=True, )
    
    valid_data = AMOuluDataset(root_folder='/mnt/training_dataset/face_dataset/Oulu_align',\
                            mode='valid',\
                            transform=transform(mode='valid'),)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=True, num_workers=4)

    # model
    # model = CDCNpp(basic_conv=Conv2d_cd, theta=0.7, num_class=30)
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    AM_model = AdMSoftmaxLoss(in_features=512, out_features=30)

    model = model.cuda()
    # model.load_state_dict(torch.load('models/model.pt'))
    # model = model.to(device[0])
    # model = nn.DataParallel(model, device_ids=device, output_device=device[0])

    AM_model = AM_model.cuda()
    # AM_model.load_state_dict(torch.load('models/AM_model.pt'))
    # AM_model = AM_model.to(device[0])
    # AM_model = nn.DataParallel(AM_model, device_ids=device, output_device=device[0])
    # print('load weight !!')

    params = list(model.parameters()) + list(AM_model.parameters())
    optimizer = optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    # # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print(f"train loader size: {len(train_loader)}")
    total_step = len(train_loader) * (args.epochs)
    lr_scheduler = PolyScheduler(
        optimizer=optimizer,
        base_lr=args.lr,
        max_steps=total_step,
        warmup_steps=0,
        last_epoch=-1
    )

    
    # #bandpass_filter_numpy = build_bandpass_filter_numpy(30, 30)  # fs, order  # 61, 64
    model_save_path = './models/model.pt'
    AM_model_save_path = './models/AM_model.pt'
    best_ACER = math.inf
    best_model_wts = copy.deepcopy(model.state_dict())
    best_AMmodel_wts = copy.deepcopy(AM_model.state_dict())
    train_idx = 0
    val_idx = 0
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        # loss_absolute = AvgrageMeter()
        # loss_contra =  AvgrageMeter()

        ###########################################
        '''                train             '''
        ###########################################
        model.train()
        AM_model.train()
        for i, (data1, data2, labels, binary) in enumerate(train_loader):
            # with torch.autograd.detect_anomaly():
            # get the inputs
            inputs1, inputs2, labels, binary = data1.cuda(), data2.cuda(), labels.cuda(), binary.cuda()
            optimizer.zero_grad()

            # forward + backward + optimize
            # feat_vector1, map_x1 = model(inputs1)
            # feat_vector2, map_x2 = model(inputs2)
            feat_vector1 = model(inputs1)
            feat_vector2 = model(inputs2)
            # self-supervised similarity
            feat_vector1 = F.normalize(feat_vector1, dim=1)
            feat_vector2 = F.normalize(feat_vector2, dim=1)
            simi_loss = torch.mean(torch.norm(feat_vector1-feat_vector2, dim=1))
            # AM-Softmax loss
            AM_loss1 = AM_model(feat_vector1, labels)
            AM_loss2 = AM_model(feat_vector2, labels)
            AM_loss = AM_loss1 + AM_loss2
            # Full loss
            loss = simi_loss + AM_loss
            # print(loss)
            # eps = 1e-6
            # if loss.isnan():
            #     print(inputs1)
            #     print(inputs2)
            #     break
                # continue
            # else: loss = loss
            # Accuracy
            # live_prob1 = AM_model._predict(feat_vector1)
            # live_prob2 = AM_model._predict(feat_vector2)
            # y_1 = torch.ones(16)
            # y_2 = torch.ones(16)
            # real_1 = live_prob1 >= 0.5 
            # real_2 = live_prob2 >= 0.5
            # real1_indices = real_1.nonzero().squeeze(1)
            # real2_indices = real_2.nonzero().squeeze(1)
            # y_1[real1_indices] = 0
            # y_2[real2_indices] = 0
            # acc1 = (y_1.cuda() == binary).float().mean()
            # acc2 = (y_2.cuda() == binary).float().mean()
            # acc = (acc1+acc2)/2
            
            
            writer.add_scalar('Loss/Train', loss, train_idx)
            # writer.add_scalar('Acc/Train', acc, train_idx)
            train_idx += 1
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            if i % args.echo_batches == 0:    # print every 50 mini-batches
                # log written
                print('epoch:%d, mini-batch:%3d, lr=%.7f, simi_loss= %.4f, AM_loss= %.4f'% (epoch + 1, i, float(lr_scheduler.get_last_lr()[0]),  simi_loss, AM_loss))
                log_file.write('epoch:%d, mini-batch:%3d, lr=%.7f, simi_loss= %.4f, AM_loss= %.4f \n' % (epoch + 1, i, float(lr_scheduler.get_last_lr()[0]),  simi_loss, AM_loss))
                log_file.flush()
        for name, param in AM_model.named_parameters():
            # writer.add_histogram(tag=name+'_grad', values=param.grad, global_step=epoch)
            writer.add_histogram(tag=name+'_data', values=param.data, global_step=epoch+1)

        # whole epoch average
        # print('epoch:%d, Train:  Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f\n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        # log_file.write('epoch:%d, Train: Absolute_Depth_loss= %.4f, Contrastive_Depth_loss= %.4f \n' % (epoch + 1, loss_absolute.avg, loss_contra.avg))
        # log_file.flush()


        if (epoch+1) % 5 == 0:    # test every 5 epochs
            model.eval()
            AM_model.eval()
            with torch.no_grad():
                ###########################################
                '''                val             '''
                ###########################################
                # val for threshold
                score_list = []
                print('------Start Validation------')
                log_file.write('------Start Validation------\n')
                log_file.flush()
                for i, (inputs, labels, binary) in enumerate(valid_loader):
                    # get the inputs
                    inputs, labels, binary = inputs, labels.cuda(), binary.cuda()
                    optimizer.zero_grad()
                    if i % 500:
                        print(f'index: {i}')
                    # forward + backward + optimize
                    for j in range(len(inputs)):
                        img_i = inputs[j].cuda()
                        # feat_vectori, map_xi = model(img_i)
                        feat_vectori = model(img_i)
                        live_prob = AM_model._predict(feat_vectori)
                        if j == 0:
                            sum_prob = live_prob
                        else:
                            sum_prob += live_prob
                    avg_prob = sum_prob / 9
                    for k in range(int(avg_prob.shape[0])):
                        live_prob = avg_prob[k]
                        score_list.append('{} {}\n'.format(live_prob, int(binary[k])))
                
                map_score_val_filename = args.log+'/'+ args.log+'_map_score_val.txt'
                with open(map_score_val_filename, 'w') as file:
                    file.writelines(score_list)  
             
                
    #             ###########################################
    #             '''                test             '''
    #             ##########################################
    #             # test for ACC
    #             test_data = LccFasdDataset(root_dir='/mnt/training_dataset/face_dataset/LCC_FASD',\
    #                             protocol='combine_all',\
    #                             transform=lcc_transform,\
    #                             get_img_path=False)
    #             dataloader_test = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=4)
    #             map_score_list = []
    #             print('------Start Test------')
    #             for i, (inputs, labels) in enumerate(dataloader_test):
    #                 # get the inputs
    #                 inputs, spoof_label = inputs.cuda(), labels.cuda()
    #                 map_score = 0.0
                    
    #                 for frame_t in range(inputs.shape[0]):
    #                     map_x, embedding, x_Block1, x_Block2, x_Block3, x_input =  model(inputs)
    #                     score_norm = torch.sum(map_x)/1024
    #                     map_score += score_norm
                    
    #                 map_score = map_score/inputs.shape[0]
    #                 map_score_list.append('{} {}\n'.format(map_score, int(spoof_label)))
    #             map_score_test_filename = args.log+'/'+ args.log+'_map_score_test.txt'
    #             with open(map_score_test_filename, 'w') as file:
    #                 file.writelines(map_score_list)    
                
    #             #############################################################     
    #             #       performance measurement both val and test
    #             #############################################################     
                val_threshold, val_ACC, val_AUC, val_APCER, val_BPCER, val_ACER = performances(map_score_val_filename)
                print('epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_AUC= %.4f, val_APCER= %.4f, val_BPCER= %.4f, val_ACER= %.4f' % (epoch + 1, val_threshold, val_ACC, val_AUC, val_APCER, val_BPCER, val_ACER))
                log_file.write('epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_AUC= %.4f, val_APCER= %.4f, val_BPCER= %.4f, val_ACER= %.4f \n' % (epoch + 1, val_threshold, val_ACC, val_AUC, val_APCER, val_BPCER, val_ACER))
                log_file.flush()
    #             print('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
    #             #print('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
                if val_ACER < best_ACER:
                    best_ACER = val_ACER
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_AMmodel_wts = copy.deepcopy(AM_model.state_dict())
                    torch.save(model.state_dict(), model_save_path)
                    torch.save(AM_model.state_dict(), AM_model_save_path)
                    print('...Saving model with ACER: {:.4f}'.format(val_ACER))
                    log_file.write('...Saving model with ACER: {:.4f} \n'.format(val_ACER))
                    log_file.flush()

    #             log_file.write('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % (epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
    #             #log_file.write('epoch:%d, Test:  test_threshold= %.4f, test_ACER_test_threshold= %.4f \n\n' % (epoch + 1, test_threshold, test_ACER_test_threshold))
    #             log_file.flush()
                writer.add_scalar('Metrics/vACER', val_ACER, val_idx)
    #             writer.add_scalar('Metrics/tACER', test_ACER, val_idx)
    #             writer.add_scalar('Metrics/tAPCER', test_APCER, val_idx)
    #             writer.add_scalar('Metrics/tBPCER', test_BPCER, val_idx)
    print('Finished Training')


  
 

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpu', type=int, default=3, help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')  
    parser.add_argument('--batchsize', type=int, default=64, help='initial batchsize')  
    # parser.add_argument('--step_size', type=int, default=100, help='how many epochs lr decays once')  # 500 
    # parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=200, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCNpp_patch", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    args = parser.parse_args()
    train_test()
