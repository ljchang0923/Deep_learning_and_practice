import torch.nn as nn
from functools import reduce

class EEGNet(nn.Module):
    def __init__(self, act_func = 'ELU'):
        super(EEGNet, self).__init__()

        ## activation
        if(act_func=='ELU'):
            self.activation = nn.ELU()
        elif(act_func=='ReLU'):
            self.activation = nn.ReLU()
        elif(act_func=='LeakyReLU'):
            self.activation = nn.LeakyReLU()
        else:
            raise BaseException("incorrect activation function")

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 61), stride=(1,1), padding=(0,30), bias=False),
            nn.BatchNorm2d(16)
        )
        self. depthwiseConv = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            self.activation,
            nn.AvgPool2d((1,4), stride=(1,4), padding=0),
            nn.Dropout(0.25)
        )
        self.separableConv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 15), stride=(1,1), padding = (0,7), bias=False),
            nn.BatchNorm2d(32),
            self.activation,
            nn.AvgPool2d((1,8), stride=(1,8), padding=0),
            nn.Dropout(0.25)
        )
        self.classifier = nn.Sequential(
            nn.Linear(736, 2, bias=True),
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        # print(f'shape before flatten: {x.size()}')
        x = x.view(-1, self.classifier[0].in_features)
        #print(f'reshape: {x.size()}')
        out = self.classifier(x)

        return out

class DeepConvNet(nn.Module):
    def __init__(self, act_func='ELU', kernel_num=[25, 50, 100, 200]):
        super(DeepConvNet, self).__init__()

        if(act_func=='ELU'):
            self.activation = nn.ELU()
        elif(act_func=='ReLU'):
            self.activation = nn.ReLU()
        elif(act_func=='LeakyReLU'):
            self.activation = nn.LeakyReLU()
        else:
            raise BaseException("incorrect activation function")

        self.kernel_num = kernel_num #[25, 50, 100, 200]

        self.conv0 = nn.Sequential(
                nn.Conv2d(1, self.kernel_num[0], kernel_size=(1,5), stride=(1,1), padding=(0,0), bias=True),
                nn.Conv2d(kernel_num[0], kernel_num[0], kernel_size=(2,1), stride=(1,1), padding=(0,0), bias=True),
                nn.BatchNorm2d(kernel_num[0]),
                self.activation,
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(0.5)
            )

        for idx in range(1, len(kernel_num)):
            setattr(self,'conv'+str(idx), nn.Sequential(
                nn.Conv2d(kernel_num[idx-1], kernel_num[idx], kernel_size=(1,5), stride=(1,1), padding=(0,0), bias=True),
                nn.BatchNorm2d(kernel_num[idx]),
                self.activation,
                nn.MaxPool2d(kernel_size=(1,2)),
                nn.Dropout(0.5)
            ))

        flatten_size = kernel_num[-1]*reduce(lambda x,_: round((x-4)/2), kernel_num, 750)
        self.classifier = nn.Sequential(
                nn.Linear(flatten_size, 2, bias=True)
            )


    def forward(self, x):

        for i in range(len(self.kernel_num)):
            x = getattr(self, 'conv' + str(i))(x)
            
        x = x.view(-1, self.classifier[0].in_features)
        output = self.classifier(x)

        return output

