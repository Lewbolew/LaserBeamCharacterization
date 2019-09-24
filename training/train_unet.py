import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from data_loader import LaserBeamData
from unet import UNet
from SegmentationModel import EESPNet_Seg
from BiSeNet import BiSeNet


from tqdm import tqdm
import time
import cv2
import os
import skimage.io as io
import numpy as np
from tensorboardX import SummaryWriter
from metrics_evaluator import PerformanceMetricsEvaluator
import torchvision.utils as vutils
from torchvision import transforms

import warnings
warnings.filterwarnings("ignore")

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return np.argmax(memory_available)

def train(model, train_loader, val_loader, optimizer, num_epochs, path_to_save_best_weights):
    model.train()

    log_softmax = nn.LogSoftmax(dim=1)# Use for NLLLoss()
    softmax = nn.Softmax(dim=1)

    criterion_nlloss = nn.NLLLoss()

    metrics_evaluator = PerformanceMetricsEvaluator()

    to_tensor = transforms.ToTensor()

    writer = SummaryWriter('runs/BiSeNet_resnet18_sum/')

    since = time.time()

    best_model_weights = model.state_dict()
    best_IoU = 0.0 
    best_val_loss = 1000000000

    curr_val_loss = 0.0
    curr_training_loss = 0.0
    curr_training_IoU = 0.0
    curr_val_IoU = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:

            if phase == 'train':
                # scheduler.step(best_val_loss)
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_IoU = 0 

            # Iterate over data.
            for imgs, masks in tqdm(data_loader):

                imgs, masks = imgs.to(device), masks.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                logits = model(imgs)
                log_softmax_logits = log_softmax(logits)
                # softmax_logits = softmax(logits)
                # loss = criterion_mseloss(softmax_logits, mask_to_encode)
                loss = criterion_nlloss(log_softmax_logits, masks)
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                softmax_logits = softmax(logits)
                collapsed_softmax_logits = np.argmax(softmax_logits.detach(), axis=1)
                prediction = np.argmax(softmax_logits[0,:, :, :].detach().cpu().numpy(), axis=0)
                empty_channel = np.zeros((1024, 1024), dtype=np.uint64)
                # final_vis_img = np.stack([prediction, masks[0], empty_channel], axis=0)
                _, mask_contours, _ = cv2.findContours(np.expand_dims(masks[0].cpu().numpy().astype(np.uint8), axis=-1),
                    cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                _, pred_contours, _ = cv2.findContours(prediction.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                # pred_contours = [cnt for cnt in pred_contours if cv2.contourArea(cnt) > 1000]
                res_img = np.copy(imgs[0])
                fof = (np.moveaxis(res_img, 0, -1)*255).astype(np.uint8).copy()
                cv2.drawContours(fof, mask_contours, -1, (0,255,0), 2)
                cv2.drawContours(fof, pred_contours, -1, (255,0,0), 2)
                if phase == 'val':
                    name = 'ValidationEpoch'
                else:
                    name = 'TrainingEpoch'
                writer.add_image('{}: {}'.format(name, str(epoch)), np.moveaxis(fof, -1, 0),epoch)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.detach().item()

                batch_IoU = 0.0
                for k in range(len(imgs)):
                    batch_IoU += metrics_evaluator.mean_IU(collapsed_softmax_logits.numpy()[k], masks.cpu().numpy()[k])
                batch_IoU /= len(imgs)
                running_IoU += batch_IoU
            epoch_loss = running_loss / len(data_loader)
            epoch_IoU = running_IoU / len(data_loader)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_IoU))
 
            # deep copy the model
            if phase == 'val' and epoch_loss < best_val_loss: # TODO add IoU
                best_val_loss = epoch_loss
                best_IoU = epoch_IoU
                best_model_weights = model.state_dict()
    
            if phase == 'val':
                # print(optimizer.param_groups[0]['lr'])
                curr_val_loss = epoch_loss
                curr_val_IoU = epoch_IoU
            else:
                curr_training_loss = epoch_loss
                curr_training_IoU = epoch_IoU

        writer.add_scalars('TrainValIoU', 
                            {'trainIoU': curr_training_IoU,
                             'validationIoU': curr_val_IoU
                            },
                            epoch
                           )
        writer.add_scalars('TrainValLoss', 
                            {'trainLoss': curr_training_loss,
                             'validationLoss': curr_val_loss
                            },
                            epoch
                           ) 
    # Saving best model
    torch.save(best_model_weights, 
        os.path.join(path_to_save_best_weights, 'BiSeNet_resnet18_sum{:2f}.pth'.format(best_val_loss)))

    # Show the timing and final statistics
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_val_loss)) # TODO add IoU




# Choose free GPU
device = torch.device("cuda:{}".format(str(get_freer_gpu())))

ROOT_DIR = 'data/'

# Create Data Loaders
partition = 'train'
ultrasound_train = LaserBeamData(ROOT_DIR, partition)
train_loader = torch.utils.data.DataLoader(ultrasound_train,
                                             batch_size=1, 
                                             shuffle=True,
                                            )
partition = 'val'
ultrasound_val = LaserBeamData(ROOT_DIR, partition, augment=False)
val_loader = torch.utils.data.DataLoader(ultrasound_val,
                                        batch_size=1,
                                        shuffle=False
                                        )
# Create model
# model = UNet((3,512,512))
# model = EESPNet_Seg(classes=2, s=2)
model = BiSeNet(2, 'resnet18')

# model.load_state_dict(torch.load('weights_unet/Unet_35_epoch_0.03_loss.pt'))
model.to(device)

# Specify optimizer and criterion
lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)
NUM_OF_EPOCHS = 100

#training
train(model, train_loader, val_loader, optimizer, NUM_OF_EPOCHS, 'weights/')
