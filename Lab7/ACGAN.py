import argparse
import os
import numpy as np
import random
import math
import json
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
import torchvision.transforms as transforms

from model import Generator, Discriminator
from evaluator import evaluation_model

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
random.seed(manualSeed)
torch.manual_seed(manualSeed)
FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", default='img/')
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--ngf", type=int, default=64, help="number of the feature map in generator")
    parser.add_argument("--ndf", type=int, default=64, help="number of the feature map in discriminator")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--n_classes", type=int, default=24, help="number of classes for dataset")
    parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
    parser.add_argument("--img_channel", type=int, default=3)
    parser.add_argument("--device", default='cuda')
    args = parser.parse_args()
    return args


class ImgDataLoader(Dataset):
    def __init__(self, args, img_path, obj_list):
        
        self.args = args
        with open(img_path, 'r') as training_file:
            train_data = json.load(training_file)
        self.train_img_name = list(train_data)
        self.train_obj = list(train_data.values()) 
        self.obj_list = obj_list
        self.img_root = args.img_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.train_img_name)

    def __getitem__(self,index):
        img = Image.open(self.img_root+self.train_img_name[index]).resize((self.args.img_size,self.args.img_size)).convert('RGB')
        img = (np.array(img)/255.0).astype(np.float32)
        img = self.transform(img)
        label = torch.zeros(24)
        for obj in self.train_obj[index]:
            label[self.obj_list[obj]] = 1

        return img, label



def get_dataloader(args):

    with open('objects.json', 'r') as object_file:
        obj_list = json.load(object_file)

    train_dataset = ImgDataLoader(args, 'train.json', obj_list)

    train_dl = DataLoader(
        dataset=train_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
    )

    with open('test.json', 'r') as test_file:
        test_list = json.load(test_file)
    
    test_label = torch.zeros(32,24)
    for idx, objs in enumerate(test_list):
        for obj in objs:
            test_label[idx, obj_list[obj]]=1

    with open('new_test.json', 'r') as new_test_file:
        new_test_list = json.load(new_test_file)
    
    new_test_label = torch.zeros(32,24)
    for idx, objs in enumerate(new_test_list):
        for obj in objs:
            new_test_label[idx, obj_list[obj]]=1


    return train_dl, test_label.long().to(args.device), new_test_label.long().to(args.device)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def train_model(gen, dis, train_dl, test_label, new_test_label, args, writer):
    optimizerG = optim.Adam(gen.parameters(), lr=args.lr, betas = (args.b1, args.b2)) 
    optimizerD = optim.Adam(dis.parameters(), lr=args.lr, betas = (args.b1, args.b2))
    criterion = nn.BCELoss()
    eval_model = evaluation_model()
    img_list = []
    niter = 0
    acc_max = 0

    for epoch in range(args.n_epochs):
        for i, (img, label) in enumerate(train_dl):
            img, label = img.to(args.device), label.to(args.device)
            batch_size = img.size(0)
            real = Variable(torch.FloatTensor(batch_size,1).fill_(0.95), requires_grad=False).to(args.device)
            fake = Variable(torch.FloatTensor(batch_size,1).fill_(0.), requires_grad=False).to(args.device)

            for _ in range(4):
                ## train real discriminator
                optimizerD.zero_grad()
                validity_real, pred_real = dis(img)
                d_real_loss = criterion(validity_real, real)
                real_class_loss = criterion(pred_real, label)
                # aux_pred, _ = eval_model.eval(img, label.long())
                # d_aux_loss = criterion(pred_real, aux_pred)

                ## train fake discriminator
                z = torch.randn(batch_size, args.latent_dim, 1, 1, device=args.device)
                gen_img = gen(z,label.long())
                validity_fake, pred_fake = dis(gen_img.detach())
                d_fake_loss = criterion(validity_fake, fake)
                # fake_class_loss = criterion(pred_fake, label)

                d_loss = d_real_loss + d_fake_loss + real_class_loss * 50
                # print("d loss: ", d_loss)
                d_loss.backward()
                optimizerD.step()

            # train generator
            
            optimizerG.zero_grad()
            # z = torch.randn(batch_size, args.latent_dim, 1, 1, device=args.device)
            gen_img = gen(z, label.long())
            validity, pred = dis(gen_img)
            aux_pred, _ = eval_model.eval(gen_img, label.long())
            g_aux_loss = criterion(pred, aux_pred)
            g_real_loss = criterion(validity, real)
            g_class_loss = criterion(pred, label)
            g_loss = g_real_loss  + g_class_loss * 50 # class loss * 100
            # print("pred: ", pred)
            # print('g real loss:', g_real_loss)
            # print('g class loss:', g_class_loss.detach().cpu().numpy())
            g_loss.backward()
            optimizerG.step()




            writer.add_scalar('train/g_loss', g_loss.item(), niter)
            writer.add_scalar('train/d_loss', d_loss.item(), niter)
            # writer.add_scalar('train/class_loss', class_loss.item(), niter)
            if i%10==0:
                print("[Epoch %d/%d] [Batch %d/%d] D loss: %.5f || G loss: %.5f || D class loss: %.5f || G class loss: %.5f"
                %(epoch+1, args.n_epochs, i, len(train_dl), d_loss.item(), g_loss.item(), real_class_loss.item()*50, g_class_loss.item()*50))
            
            niter+=1

        fix_z = torch.randn(32, args.latent_dim, 1, 1, device=args.device)
        with torch.no_grad():
            fake = gen(fix_z, test_label) 
            fake_new = gen(fix_z, new_test_label)
        upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        fake = upsample(fake)
        fake_new = upsample(fake_new)
        # fake = inv_transform(fake)
        _, acc = eval_model.eval(fake, test_label)
        _, acc_new = eval_model.eval(fake_new, new_test_label)
        writer.add_scalar('test/acc', acc, epoch)
        writer.add_scalar('test/new_acc', acc_new, epoch)
        print("generated img acc: %.3f || new test img acc: %.3f"%(acc, acc_new))

        if (acc + acc_new)/2 > acc_max:
            acc_max = (acc + acc_new)/2
            torch.save(gen.state_dict(), 'model/generator.pth')
            plt.axis("off")
            plt.title(f"best Images (acc:{acc})")
            plt.imshow(np.transpose(vutils.make_grid(fake.detach().cpu(), padding=2, normalize=True),(1,2,0)))
            plt.savefig(f'best_image.png')
            plt.clf()
            img_list.append([vutils.make_grid(fake.detach().cpu(), padding=2, normalize=True), acc])
            plt.axis("off")
            # for new test data
            plt.title(f"best Images (acc:{acc_new})")
            plt.imshow(np.transpose(vutils.make_grid(fake_new.detach().cpu(), padding=2, normalize=True),(1,2,0)))
            plt.savefig(f'best_image_new.png')
            plt.clf()
            img_list.append([vutils.make_grid(fake_new.detach().cpu(), padding=2, normalize=True), acc_new])

        if epoch%10==0:
            plt.axis("off")
            plt.title(f"Fake Images (acc:{acc})")
            plt.imshow(np.transpose(vutils.make_grid(fake.detach().cpu(), padding=2, normalize=True),(1,2,0)))
            plt.savefig(f'gen_img{epoch}.png')
            plt.clf()
            # for new test data
            plt.axis("off")
            plt.title(f"Fake Images (acc:{acc_new})")
            plt.imshow(np.transpose(vutils.make_grid(fake_new.detach().cpu(), padding=2, normalize=True),(1,2,0)))
            plt.savefig(f'gen_img_new{epoch}.png')
            plt.clf()
            

    return img_list


def main():
    args = parse_args()
    writer = SummaryWriter('logs/')
    train_dl, test_label, new_test_label = get_dataloader(args)
    gen = Generator(args).to(args.device)
    gen.apply(weights_init)
    dis = Discriminator(args).to(args.device)
    dis.apply(weights_init)
    print("gen: ",gen)
    print("dis: ",dis)
    gen_img = train_model(gen, dis, train_dl, test_label, new_test_label, args, writer)

    for i in range(len(gen_img)):
        plt.axis("off")
        plt.title("Fake Images (acc: %.3f)"%(gen_img[i][1]))
        plt.imshow(np.transpose(gen_img[i][0],(1,2,0)))
        plt.savefig(f'fake{i}.png')
        plt.clf()
    
if __name__ == '__main__':
    main()
