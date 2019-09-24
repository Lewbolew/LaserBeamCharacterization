import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from data_loader import LaserBeamData
from shape_net import ShapeUNet
# from SR_Unet import SH_UNet
# from SR_UnetOriginal import SH_UNet
from SR_ESPNet_one_output import SH_UNet
from unet import UNet
from SegmentationModel import EESPNet_Seg
from r2_unet import R2U_Net, R2AttU_Net
# from BiSeNet import BiSeNet

# from SR_Unet import SH_UNet
from tqdm import tqdm
import time
import numpy as np
import os
import cv2
import skimage.io as io
import numpy as np
from metrics_evaluator import PerformanceMetricsEvaluator
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from torchvision.utils import save_image, make_grid
from torchvision import transforms
# from sklearn.metrics import jaccard_similarity_score
import warnings


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return np.argmax(memory_available)

# Data Loading
ROOT_DIR = 'data/'
partition = 'test'
ultrasound_test = LaserBeamData(ROOT_DIR, partition, augment=False)
test_loader = torch.utils.data.DataLoader(ultrasound_test,
                                        batch_size=1,
                                        shuffle=False
                                        )
# Model Creation
device = torch.device("cuda:{}".format(str(get_freer_gpu())))
PATH_TO_THE_WEIGHTS = 'weights/Unet0.004769.pth'
# model = UNet((1,512,512))



# model = SH_UNet()
model = EESPNet_Seg(classes=2, s=2)
# model = R2AttU_Net(img_ch=3, output_ch=2)
# model = BiSeNet(2, 'resnet101')
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
# model.load_state_dict(torch.load(PATH_TO_THE_WEIGHTS))
# model.to(device)
# model.eval()
# # Evaluation Techniques
# metrics_evaluator = PerformanceMetricsEvaluator()

# # Evaluation
# iou_of_the_model = 0
# mean_acc_of_the_model = 0

# softmax = nn.Softmax(dim=1)

# temporary_counter = 0

# with torch.no_grad():
#     unique_ind = 0
#     to_tensor = transforms.ToTensor()

#     for (test_imgs, test_masks) in tqdm(test_loader):
#         mask_to_encode = (np.arange(2) == test_masks.numpy()[...,None]).astype(float)
#         mask_to_encode = torch.from_numpy(np.moveaxis(mask_to_encode, 3, 1)).float().to(device)

#         test_imgs = test_imgs.to(device)
#         test_masks = test_masks.to(device)
#         mask_to_encode = mask_to_encode.to(device)

#         # unet_prediction = model(test_imgs)
#         # np.save('logits.npy', logits)
#         # temporary_counter+=1
#         # if temporary_counter > 2:
#         #     break
#         unet_prediction, shape_net_encoded_prediction, shape_net_final_prediction = model(test_imgs)

#         softmax_logits = softmax(shape_net_final_prediction)
#         collapsed_softmax_logits = np.argmax(softmax_logits.detach(), axis=1)

#         collapsed_softmax_logits[collapsed_softmax_logits==1] = 255
        
#         # save_image(collapsed_softmax_logits, 'results/shape_esp_network_without/{}.png'.format(unique_ind))

#         # save_image(test_imgs[0], 'results/x/{}.png'.format(unique_ind))
#         # save_image(test_masks[0], 'results/y/{}.png'.format(unique_ind))

#         unique_ind+=1

#         softmax_logits_unet = softmax(unet_prediction)
#         collapsed_softmax_unet = np.argmax(softmax_logits_unet.detach(), axis=1)


#         # for i in range(len(test_imgs)):
#         #     rgb_prediction = collapsed_softmax_logits[i].repeat(3, 1, 1).float()
#         #     rgb_prediction = np.moveaxis(rgb_prediction.numpy(), 0, -1)
#         #     converted_img = img_to_visible(rgb_prediction)
#         #     converted_img = to_tensor(converted_img)

#         #     # rgb_unet_prediction = collapsed_softmax_unet[i].repeat(3, 1, 1).float()
#         #     # rgb_unet_prediction = np.moveaxis(rgb_unet_prediction.numpy(), 0, -1)
#         #     # converted_img_unet = img_to_visible(rgb_unet_prediction)
#         #     # converted_img_unet = to_tensor(converted_img_unet)

#         #     masks_changed = test_masks[i].detach().cpu()
#         #     masks_changed = masks_changed.repeat(3,1,1).float()
#         #     masks_changed = np.moveaxis(masks_changed.numpy(), 0, -1)
#         #     masks_changed = img_to_visible(masks_changed)
#         #     masks_changed = to_tensor(masks_changed)
#         #     changed_imgs = torch.cat([test_imgs[i],test_imgs[i],test_imgs[i]]).detach().cpu()

#         #     changed_imgs = make_grid([changed_imgs], normalize=True, range=(0,255))
#         #     third_tensor = torch.cat((changed_imgs, masks_changed, converted_img), -1)
#         #     # third_tensor = torch.cat((changed_imgs, masks_changed, converted_img_unet, converted_img), -1)

#         # #     save_image(third_tensor, 'predictions/unet/{}'.format(str(unique_ind))+'.png')
#         #     unique_ind+=1





#         batch_iou = 0
#         batch_mean_acc = 0
#         for k in range(len(test_imgs)):
#             batch_iou += metrics_evaluator.mean_IU(collapsed_softmax_logits.numpy()[k], test_masks.cpu().numpy()[k])
#             batch_mean_acc += metrics_evaluator.mean_accuracy(collapsed_softmax_logits.numpy()[k], test_masks.cpu().numpy()[k])
#         batch_iou /= len(test_imgs)
#         batch_mean_acc /= len(test_imgs)
#         mean_acc_of_the_model += batch_mean_acc
#         iou_of_the_model += batch_iou

#     mean_acc_of_the_model /= len(test_loader)
#     iou_of_the_model /= len(test_loader)

#     print('IoU: ', iou_of_the_model)
#     print('Mean Accuracy: ', mean_acc_of_the_model)