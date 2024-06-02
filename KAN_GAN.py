import random
import numpy as np
from tqdm import tqdm
import os
import PIL.Image as PImage

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torchvision.utils as utils

from efficient_kan import KAN

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

def load_MNIST(config):
    # Load MNIST
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    trainset = torchvision.datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )
    valset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    trainloader = DataLoader(trainset, batch_size=config['BATCH_SIZE'], shuffle=True)
    valloader = DataLoader(valset, batch_size=config['BATCH_SIZE'], shuffle=False)
    
    return trainloader,valloader

def build_model(config):
    generator = KAN([config['LATENT_SIZE'],config['HIDDEN_SIZE'],config['HIDDEN_SIZE'],config['HIDDEN_SIZE'],config['IMAGE_SIZE']], grid_size=5, spline_order=3)
    discriminator = KAN([config['IMAGE_SIZE'],config['HIDDEN_SIZE'],config['HIDDEN_SIZE'],config['HIDDEN_SIZE'],config['OUTPUT_SIZE']], grid_size=5, spline_order=3)
    
    return generator,discriminator
    
def build_optimizer(generator,discriminator,config):
    optimizer_G = optim.Adam(generator.parameters(), lr = config['LEARNING_RATE'],betas=(0.9,0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr = config['LEARNING_RATE'],betas=(0.9,0.999))
    
    return optimizer_G,optimizer_D
    
def main():
    config = {
    'IMAGE_SIZE':784,
    'LATENT_SIZE':100,
    'HIDDEN_SIZE':256,
    'OUTPUT_SIZE':2,
    'EPOCHS':50,
    'LEARNING_RATE':0.00002,
    'BATCH_SIZE':256,
    'DEVICE': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'SEED':42
    }
    
    #seed_everything(config['SEED'])
    trainloader,valloader = load_MNIST(config)
    generator,discriminator = build_model(config)
    optimizer_G,optimizer_D = build_optimizer(generator,discriminator,config)
    criterion = nn.CrossEntropyLoss()
    device = config['DEVICE']
    
    #Train
    generator.to(device)
    discriminator.to(device)
    for epoch in range(1,config['EPOCHS']+1):
        generator.train()
        discriminator.train()
        with tqdm(trainloader) as pbar:
            for i, (images, labels) in enumerate(pbar):
                
                batch_size = images.size()[0]
                images = images.view(batch_size,-1).to(device)
                
                real_label = torch.ones((batch_size),device=device,dtype=torch.long)
                fake_label = torch.zeros((batch_size),device=device,dtype=torch.long)
                
                # noise sample
                z = torch.randn(batch_size, config['LATENT_SIZE'],device=device)
                
                G_x = generator(z)
                G_x = torch.nn.functional.tanh(G_x)

                # discriminator 학습 / G 고정
                D_x = discriminator(images)
                D_loss = criterion(D_x,real_label)
                
                D_z = discriminator(G_x.detach())
                Z_loss = criterion(D_z,fake_label)
                
                total_loss = D_loss + Z_loss
                
                optimizer_D.zero_grad()
                total_loss.backward()
                optimizer_D.step()
                
                # Generator 학습 / D 고정
                D_z = discriminator(G_x)
                G_loss = criterion(D_z,real_label)
                
                optimizer_G.zero_grad()
                G_loss.backward()
                optimizer_G.step()
        
        if epoch % 10 == 0:
            generator.eval()
            z = torch.randn(16, config['LATENT_SIZE'],device=device)
            generate_img = torch.nn.functional.tanh(generator(z))
            generate_img = generate_img.reshape((-1,28,28)).unsqueeze(1)
            
            generate_img = utils.make_grid(generate_img)
            generate_img = (generate_img+1)/2
            generate_img = generate_img.permute(1, 2, 0).mul_(255).cpu().numpy()
            pimg = PImage.fromarray(generate_img.astype(np.uint8))
            pimg.save(f'/home/shu/Desktop/Yongjin/GenAI/Project/KAN/vis/KAN_GAN{epoch}.png','png')
    
if __name__ == '__main__':
    main()