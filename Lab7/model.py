import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(args.n_classes, 1)
        self.fc1 = nn.Linear(124, args.ngf*16)
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.ngf * 16, args.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( args.ngf, args.img_channel, 4, 1, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z, label):
        label = label.unsqueeze(2).unsqueeze(3)
        # label_emb = self.label_emb(label).unsqueeze(-1)
        latent = torch.cat((z, label), 1)
        latent = latent.view(-1, 124)
        x = self.fc1(latent)
        x = x.view(-1,x.size(1),1,1)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.img_channel, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
             # state size. (ndf*8) x 4 x 4
        )
        self.dis = nn.Sequential(
            nn.Linear(args.ndf*8*4,1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(args.ndf*8*4,args.n_classes),
            nn.Sigmoid() 
        )
                  

    def forward(self, x):
        x = self.main(x)
        latent = torch.flatten(x,1)
        # print("latent: ", latent.size())
        output = self.dis(latent)
        pred = self.classifier(latent)
        return output, pred