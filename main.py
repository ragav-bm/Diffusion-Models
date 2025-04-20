import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from pathlib import Path
import os
from model import Unet
from diffusion import Diffusion, linear_beta_schedule,cosine_beta_schedule,sigmoid_beta_schedule
from torchvision.utils import save_image
import argparse
import matplotlib.pyplot as plt
from datasets import load_dataset
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Train a neural network to diffuse images')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--timesteps', type=int, default=1000, help='number of timesteps for diffusion model (default: 100)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate (default: 0.003)')
    
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    
    parser.add_argument('--log_interval', type=int, default=100, help='how many batches to wait before logging training status')
    parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument('--run_name', type=str, default="DDPM")
    parser.add_argument('--dry_run', action='store_true', default=False, help='quickly check a single pass')
    return parser.parse_args()

def sample_and_save_images(n_images, diffusor, model, device, store_path,reverse_transform,class_lbl = 0,class_free_guidance = False,w=0):
    
    if class_lbl ==11:
        y = None
    else:
        print("helooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo")
        y = class_lbl
    images = []
    for i in range(n_images):
        x_t = diffusor.sample(model, diffusor.img_size, y = y,class_free_guidance=class_free_guidance,w = w).to(device)
        
        image = reverse_transform(x_t[-1].detach().cpu())
        
        images.append(image)
    img_width, img_height = images[0].size
    grid_image = Image.new("RGB", (img_width * 3, img_height * 3))  
    for idx, img in enumerate(images):
        grid_image.paste(img, ((idx % 3) * img.width, (idx // 3) * img.height))
    
    grid_image.save("/proj/ciptmp/ur03ocab/model_testing_ex2/sample_50_epoch_1000_timesteps_linear/image_grid1000.png")
    print(f"Saved image grid")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def test(model, testloader, diffusor, device, args):
    
    image_size= diffusor.img_size
    batch_size = args.batch_size
    timesteps = args.timesteps
    total_loss =0
    batch_count =0
    generate_image = []
    pbar = tqdm(testloader)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, y= labels, loss_type="l2")
        total_loss = total_loss + loss.item()
        batch_count = batch_count + 1
        
        
        
        
    average_loss = total_loss / batch_count
    return average_loss , generate_image
def train(model, trainloader, optimizer, diffusor, epoch, device, args):
    batch_size = args.batch_size
    timesteps = args.timesteps
    loss_values = []
    epoch_losses = []
    epoch_losses_0 =[]
    pbar = tqdm(trainloader)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        t = torch.randint(0, timesteps, (len(images),), device=device).long()
        loss = diffusor.p_losses(model, images, t, y = labels,loss_type="l2")
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        if epoch == args.epochs - 1:
            epoch_losses.append(loss.item())
        if epoch == 0:
            epoch_losses_0.append(loss.item())
        if step % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(images), len(trainloader.dataset),
                100. * step / len(trainloader), loss.item()))
        if args.dry_run:
            break
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label="Training Loss")
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('/proj/ciptmp/ur03ocab/model_testing_ex2/loss-linear_1000t.png')
    print("loss_epoch saved")
    if epoch == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_losses_0, label="first Epoch Loss")
        plt.xlabel('Timestep')
        plt.ylabel('Loss')
        plt.title(f'Loss vs Timestep for Epoch 0')
        plt.legend()
        plt.grid(True)
        plt.savefig('/proj/ciptmp/ur03ocab/model_testing_ex2/loss_first_epoch_linear_1000_timesteps.png')
        print("Last epoch losses for timesteps saved")
    if epoch == args.epochs - 1:
        plt.figure(figsize=(10, 6))
        plt.plot(epoch_losses, label="Last Epoch Loss")
        plt.xlabel('Timestep')
        plt.ylabel('Loss')
        plt.title(f'Loss vs Timestep for last Epoch ')
        plt.legend()
        plt.grid(True)
        plt.savefig('/proj/ciptmp/ur03ocab/model_testing_ex2/loss_last_epoch_linear_1000_timesteps.png')
        print("Last epoch losses for timesteps saved")
def run(args):
    timesteps = args.timesteps
    image_size = 32  
    channels = 3
    epochs = args.epochs
    batch_size = args.batch_size
    device = "cuda" if not args.no_cuda and torch.cuda.is_available() else "cpu"
    class_lbl = 0
    class_free_guidance = False
    p_uncond = 0.1
    num_classes = 10
    w = 0.1
    s_limit =10
    model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,),class_free_guidance=class_free_guidance,p_uncond =p_uncond,num_classes=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    my_scheduler = lambda x: linear_beta_schedule(0.0001, 0.02, x)
    
    
    diffusor = Diffusion(timesteps, my_scheduler, image_size, device)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    transform = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),    
        transforms.Lambda(lambda t: (t * 2) - 1)   
    ])
    reverse_transform = Compose([
        Lambda(lambda t: (t.clamp(-1, 1) + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),  
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.numpy().astype(np.uint8)),
        ToPILImage(),
    ])
    dataset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=True, transform=transform)
    
    
    trainset, valset = torch.utils.data.random_split(dataset, [int(len(dataset) * 0.9), len(dataset) - int(len(dataset) * 0.9)])
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    
    testset = datasets.CIFAR10('/proj/aimi-adl/CIFAR10/', download=True, train=False, transform=transform)
    
    
    testloader = DataLoader(testset, batch_size=int(batch_size/2), shuffle=True)
    
    
    
    
    for epoch in range(epochs):
        train(model, trainloader, optimizer, diffusor, epoch, device, args)
        test(model, valloader, diffusor, device, args)
    test(model, testloader, diffusor, device, args)
    save_path = "/home/cip/nf2024/ur03ocab/Desktop/AdvancedDeepLearning/ex_2/"  
    n_images = 16
    
    torch.save(model.state_dict(),
               os.path.join("/proj/ciptmp/ur03ocab/model_testing_ex2/", args.run_name,
                            f"sample_50_epoch_1000_timesteps_linear_loss.pt"))
    sample_and_save_images(n_images, diffusor, model, device, save_path,reverse_transform,class_lbl,class_free_guidance,w)
if __name__ == '__main__':
    args = parse_args()
    
    run(args)
