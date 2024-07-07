import numpy as np
from torch.autograd import Variable
import torchvision
import torch
import time
import model_init
import gc
import streamlit as st


NORMALIZE_MEAN= torchvision.models.VGG19_Weights.IMAGENET1K_V1.transforms().mean
NORMALIZE_STD= torchvision.models.VGG19_Weights.IMAGENET1K_V1.transforms().std

# if torch.cuda.is_available():
#     device = 'cuda'
#     print("CUDA is available. Using GPU.")
# else:
#     device = 'cpu'
#     print("CUDA is not available. Using CPU.")

device='cuda'

def pre_process_images(style_img, content_img):
    if((content_img.shape[0] * content_img.shape[1]) <( 641*641) ):
        resize_shape=content_img.shape[:-1]
    else:
        resize_shape=(641,641)
    
    transforms=torchvision.transforms.Compose([  torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Resize(resize_shape),
                                           torchvision.transforms.Normalize(NORMALIZE_MEAN,NORMALIZE_STD)
                                           ])
    style_img=transforms(style_img)
    content_img=transforms(content_img)
    style_img.requires_grad=False
    content_img.requires_grad=False
    return style_img, content_img

def get_gram_matrix(feature_maps):
   c,h,w = feature_maps.shape
   fm=feature_maps.reshape(c,h*w)
   return torch.mm(fm, torch.transpose(fm,0,1))




def print_gpu_memory_usage(step):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    print(f"{step}:")
    print(f"  Allocated: {allocated / 1024**2:.2f} MB")
    print(f"  Cached: {reserved / 1024**2:.2f} MB")

def Neural_Style_Transfer(style_img, content_img, epochs,update_progress):

    style_img, content_img= pre_process_images(style_img, content_img)
    print("Preprocessing Ends")
    # print_gpu_memory_usage("Initial")

    vgg_model=model_init.vgg19(['conv1_1','conv2_1','conv3_1','conv4_1','conv5_1'],'conv4_1')
    for param in vgg_model.parameters():
        param.requires_grad=False
    vgg_model.to(device)
    # print_gpu_memory_usage("After loading VGG19")
    style_img=style_img.to(device)
    print(style_img.shape)
    content_img=content_img.to(device)
    noisy_img=Variable(content_img.clone().to(device),requires_grad=True)
    print("Model and Images on ",device )
    train_neural_style_transfer_LBFGS(vgg_model, style_img,content_img, noisy_img, epochs,update_progress)
    output_img=noisy_img.to('cpu')
    output_img = output_img*torch.tensor(NORMALIZE_STD).view(3, 1, 1) + torch.tensor(NORMALIZE_MEAN).view(3, 1, 1)
    output_img = torch.clamp(output_img, 0, 1)
    
    # Release GPU memory and clear the cache
    del style_img, content_img, noisy_img, vgg_model
    torch.cuda.empty_cache()
    gc.collect()
    
    return torchvision.transforms.functional.to_pil_image(output_img)


def train_neural_style_transfer_LBFGS(model, style_img, content_img, noisy_img,epoch,update_progress):
    optimizer=torch.optim.LBFGS((noisy_img,), lr=0.9)
    log_loss=[0,0,0]
    def closure():
        optimizer.zero_grad()
        noisy_img_output=model(noisy_img)
        style_img_output= model(style_img)
        content_img_output=model(content_img)
        
        c,h,w=noisy_img_output['style_feature_map']['conv3_1'].shape
        noisy_fm=noisy_img_output['style_feature_map']['conv3_1'].reshape(c,h*w)
        content_fm=content_img_output['style_feature_map']['conv3_1'].reshape(c,h*w)
        
        content_loss=torch.nn.MSELoss()(noisy_fm,content_fm)
        
        noisy_gm={}
        style_gm={}
        style_loss=0
        for x in noisy_img_output['style_feature_map']:
            noisy_gm[x]=get_gram_matrix(noisy_img_output['style_feature_map'][x]) 
        
        for y in style_img_output['style_feature_map']:
            style_gm[y]=get_gram_matrix(style_img_output['style_feature_map'][y])
        
        loss=0
        for x in noisy_gm:
            c,h,w=noisy_img_output['style_feature_map'][x].shape
            layer_loss=(1/20) * torch.sum(((noisy_gm[x] - style_gm[x])/(c*h*w))**2)
            style_loss+=layer_loss
        
        loss=0*content_loss + 1e10*style_loss
        log_loss[0]=loss.detach().item()
        log_loss[1]=style_loss.detach().item()
        log_loss[2]=content_loss.detach().item()

        total_var_loss=torch.sqrt((torch.nn.MSELoss()(noisy_img[:,:,:-1], noisy_img[:,:,1:]) + torch.nn.MSELoss()(noisy_img[:,:-1,:], noisy_img[:,1:,:])))
        loss+=1e3*total_var_loss
        
        loss.backward()
        return loss
    for i in range(epoch):
        optimizer.step(closure)
        print("Epoch: ",i, " Loss: ",log_loss[0], ' style_loss: ', log_loss[1], ' content_loss: ',log_loss[2])
        intermediate_output=noisy_img.to('cpu')
        intermediate_output = intermediate_output*torch.tensor(NORMALIZE_STD).view(3, 1, 1) + torch.tensor(NORMALIZE_MEAN).view(3, 1, 1)
        intermediate_output = torch.clamp(intermediate_output, 0, 1)
        intermediate_output=torchvision.transforms.functional.to_pil_image(intermediate_output)
        update_progress(i+1,intermediate_output)




        
    
    