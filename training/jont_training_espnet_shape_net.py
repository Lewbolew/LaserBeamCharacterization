import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from data_loader import LaserBeamData
from SegmentationModel import EESPNet_Seg
from shape_net import ShapeUNet


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

def train_epoch(model, train_dataloader,optimizer):
    model.train()

    log_softmax = nn.LogSoftmax(dim=1)
    criterion_nlloss = nn.NLLLoss()

    print("Training Unet: ")
    average_loss = 0.0
    for imgs, masks in tqdm(train_dataloader):

        imgs, masks = imgs.to(device), masks.to(device)
        
        logits, logits_2 = model(imgs)
        log_logits = log_softmax(logits).to(device)
        log_logits_2 = log_softmax(logits_2).to(device)

        loss = criterion_nlloss(log_logits, masks) + criterion_nlloss(log_logits_2, masks)
        average_loss+=loss

        # zero gradients
        optimizer.zero_grad()
        # backward pass
        loss.backward()
        optimizer.step()
    average_loss /= len(train_dataloader)
    print("Unet training loss: {:f}".format(average_loss))

def train_network_on_top_of_other(model, train_loader, val_loader, optimizer, 
                                  unet, unet_data_loader, unet_optim, num_epochs, 
                                  path_to_save_best_weights):

    model.train()

    log_softmax = nn.LogSoftmax(dim=1)# Use for NLLLoss()
    softmax = nn.Softmax(dim=1)

    to_tensor = transforms.ToTensor()

    metrics_evaluator = PerformanceMetricsEvaluator()

    criterion_nlloss = nn.NLLLoss()
    criterion_mseloss = nn.MSELoss(size_average=False)
    writer = SummaryWriter('runs/pretrained_shape_espnet')

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
                if epoch % 5 == 0:
                    unet.train()
                    for i in range(3):
                        train_epoch(unet, unet_data_loader,unet_optim)
                unet.eval()
                model.train()
                data_loader = train_loader
            else:
                model.eval()
                data_loader = val_loader

            running_loss = 0.0
            running_IoU = 0 

            # Iterate over data.
            ind = 0
            for imgs, masks in tqdm(data_loader):

                mask_to_encode = masks.numpy()
                mask_to_encode = (np.arange(2) == mask_to_encode[...,None]).astype(float)
                mask_to_encode = torch.from_numpy(np.moveaxis(mask_to_encode, 3, 1)).float().to(device)

                imgs = imgs.to(device)
                masks = masks.to(device)
                mask_to_encode = mask_to_encode.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase == 'train':

                    unet_output = unet(imgs).detach()
                    softmax_unet_output = softmax(unet_output).detach()
                    logits, encoded_shape = model(softmax_unet_output)
                    _, encoded_mask = model(mask_to_encode)

                    log_softmax_logits = log_softmax(logits)
                    softmax_logits = softmax(logits)
                    first_term = criterion_mseloss(softmax_logits, softmax_unet_output)
                    second_term = criterion_mseloss(encoded_shape, encoded_mask)
                    lambda_1 = 0.5
                    loss = first_term + lambda_1*second_term
                    print("First term: ", first_term, "Second term: ", lambda_1*second_term)
                else:
                    with torch.no_grad():
                        unet_output = unet(imgs).detach()
                        softmax_unet_output = softmax(unet_output).detach()
                        logits, encoded_shape = model(softmax_unet_output)
                        _, encoded_mask = model(mask_to_encode)

                        log_softmax_logits = log_softmax(logits)
                        softmax_logits = softmax(logits)
                        first_term = criterion_mseloss(softmax_logits, softmax_unet_output)
                        second_term = criterion_mseloss(encoded_shape, encoded_mask)
                        lambda_1 = 0.5
                        loss = first_term + lambda_1*second_term
                        print("First term: ", first_term, "Second term: ", lambda_1*second_term)
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                collapsed_softmax_logits = np.argmax(softmax_logits.detach(), axis=1)
                collapsed_softmax_unet = np.argmax(softmax_unet_output.detach(), axis=1)
                softmax_segm_network = softmax(unet_output)

                prediction = np.argmax(softmax_logits[0,:, :, :].detach().cpu().numpy(), axis=0)
                prediction_segm_net = np.argmax(softmax_segm_network[0,:, :, :].detach().cpu().numpy(), axis=0)

                empty_channel = np.zeros((1024, 1024), dtype=np.uint64)
                # final_vis_img = np.stack([prediction, masks[0], empty_channel], axis=0)
                _, mask_contours, _ = cv2.findContours(np.expand_dims(masks[0].cpu().numpy().astype(np.uint8), axis=-1),
                    cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                _, pred_contours, _ = cv2.findContours(prediction.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                _, pred_segm_net_contours, _ = cv2.findContours(prediction_segm_net.astype(np.uint8), 
                                                                cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                # pred_contours = [cnt for cnt in pred_contours if cv2.contourArea(cnt) > 1000]
                res_img = np.copy(imgs[0])
                fof = (np.moveaxis(res_img, 0, -1)*255).astype(np.uint8).copy()
                cv2.drawContours(fof, mask_contours, -1, (0,255,0), 2)
                cv2.drawContours(fof, pred_contours, -1, (255,0,0), 2)
                cv2.drawContours(fof, pred_segm_net_contours, -1, (0,0,255), 1)

                if phase == 'val':
                    name = 'ValidationEpoch'
                else:
                    name = 'TrainingEpoch'
                writer.add_image('{}: {}'.format(name, str(epoch)), np.moveaxis(fof, -1, 0),epoch)

                # statistics
                running_loss += loss.detach().item()
                batch_IoU = 0.0
                for k in range(len(imgs)):
                    batch_IoU += metrics_evaluator.mean_IU(collapsed_softmax_logits.numpy()[k], masks.cpu().numpy()[k])
                batch_IoU /= len(imgs)
                running_IoU+=batch_IoU

                ind+=1
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
        os.path.join(path_to_save_best_weights, 'pretrained_shape_espnet{:2f}.pth'.format(best_val_loss)))

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
partition = 'train'
unet_train = LaserBeamData(ROOT_DIR, partition)
unet_loader = torch.utils.data.DataLoader(unet_train,
                                             batch_size=1, 
                                             shuffle=True,
                                            )

# Create model
model = ShapeUNet((2,512, 512))
unet = EESPNet_Seg(classes=2, s=2)
model.to(device)
unet.to(device)

lr = 1e-4
optimizer = Adam(model.parameters(), lr=lr)
NUM_OF_EPOCHS = 40

lr1 = 1e-4
unet_optim = Adam(unet.parameters(), lr=lr1)

train_network_on_top_of_other(model, train_loader, val_loader, optimizer, 
                              unet, unet_loader, unet_optim, NUM_OF_EPOCHS, 
                              'weights/')
