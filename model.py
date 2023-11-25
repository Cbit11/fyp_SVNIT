import torch
import torch.nn as nn
import torch.nn.functional as F

##################### ESTIMATION BLOCK #####################


class Estimation_Block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels=in_c, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=out_c, kernel_size=3, stride=1, padding=1)
        
    def forward(self,x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        k = F.relu(self.conv5(cat3))
        return k
    
##################### AOD BLOCK #######################

class AOD_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_estimate = Estimation_Block(3,1)
        self.b_estimate = Estimation_Block(3,1)
        
    def forward(self, features, x):
        k = self.k_estimate(features)
        b = torch.mean(self.b_estimate(features))
        out = k * x - k + b
        return F.tanh(out)
    
#################### DEHAZE MODEL #######################

class Dehaze_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.aod_b = AOD_Block()
        self.aod_g = AOD_Block()
        self.aod_r = AOD_Block()
    
    def forward(self,x):
        b,g,r = x[:,0:1,:,:], x[:,1:2,:,:], x[:,2:3,:,:]
        b,g,r = self.aod_b(x,b), self.aod_g(x,g), self.aod_r(x,r)
        return torch.cat((b,g,r),1)
    

#################### RESTORATION MODULE #####################

def conv(in_channels, out_channels,kernel_size, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding= (kernel_size//2))
class _DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.ups = nn.ModuleList()
        self.dwns = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNet
        for feature in features:
            self.dwns.append(_DoubleConv(in_channels, feature))
            in_channels = feature
        self.bottleneck = _DoubleConv(features[-1], features[-1] * 2)

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(_DoubleConv(feature * 2, feature))

        self.out = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Down part of UNet
        for down in self.dwns:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Up part of UNet
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x,
                    size=skip_connection.shape[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        return self.out(x)

####################### MPRNET ##################################

class sam(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(sam,self).__init__()
        self.conv1= conv(in_channels, out_channels, 3)
        self.conv2= nn.Sequential(
            conv(in_channels, 3, 3),
            conv(3, out_channels, 3),
            nn.Sigmoid()
        )
    def forward(self, x):
        x1= self.conv1(x)
        x2= self.conv2(x)
        x3= x1*x2
        return x3+x
class MPRnet(nn.Module):
    def __init__(self):
        super(MPRnet,self).__init__()
        self.act= nn.PReLU()
        self.conv1= conv(1,1,kernel_size=3)
        self.conv2= conv(1,1,kernel_size=3)
        self.conv3= conv(1,1,kernel_size=3)
        self.conv4= conv(1,1,kernel_size=3)
        self.conv5= conv(1,1,kernel_size=3)
        self.conv6= conv(1,1,kernel_size=3)
        self.unet_r= UNet(1,1)
        self.unet_g= UNet(1,1)
        self.unet_b= UNet(1,1)
        self.sam_r= sam(1,1)
        self.sam_g= sam(1,1)
        self.sam_b= sam(1,1)
        self.conv7= conv(3,3,3)
        self.conv8= conv(3,3,3)
        self.unet= UNet(3,3)
    def forward(self, x):
        r= x[:,0,:,:].unsqueeze(1)
        g= x[:,1,:,:].unsqueeze(1)
        b= x[:,2,:,:].unsqueeze(1)
        ################## RED PART ###################
        r1= self.act(self.conv1(r))
        r1= self.conv2(r1)
        r1= self.unet_r(r1)
        r1= self.sam_r(r1)
        ################# GREEN PART ####################
        g1= self.act(self.conv3(g))
        g1= self.conv4(g1)
        g1= self.unet_g(g1)
        g1= self.sam_g(g1)
        ################# BLUE PART #####################
        b1= self.act(self.conv5(b))
        b1= self.conv6(b1)
        b1= self.unet_b(b1)
        b1= self.sam_b(b1)

        x1 = torch.cat((r1,g1,b1), dim = 1)
        x1= self.act(self.conv7(x1))
        out = self.conv8(x1)
        return out
    
###################### FINAL NETWORK #####################

class network(nn.Module):
    def __init__(self):
        super().__init__()
        self.dehaze_model= Dehaze_model()
        self.mpr= MPRnet()
        self.unet= UNet(6,3)
    def forward(self, x,batch_size,HEIGHT, WIDTH):
        x1= self.dehaze_model(x)
        out = self.mpr(x1)
#         print(out.shape)
        noise = torch.rand(batch_size,3, HEIGHT, WIDTH)
        out = torch.cat((out,noise), dim =1)
#         print(out.shape)
        out = self.unet(out)
        return out