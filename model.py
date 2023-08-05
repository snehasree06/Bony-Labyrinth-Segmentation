import torch
from torch import nn
from torch.nn import functional as F


class kiunet(nn.Module):
    
    def __init__(self,in_channels=1,n_classes=1):
        super(kiunet, self).__init__()
        

        self.encoder1 = nn.Conv2d(in_channels, 16, 3, stride=1, padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB 
        self.en1_bn = nn.BatchNorm2d(16)
        self.encoder2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)  
        self.en2_bn = nn.BatchNorm2d(32)
        self.encoder3=   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.en3_bn = nn.BatchNorm2d(64)

        self.decoder1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)   
        self.de1_bn = nn.BatchNorm2d(32)
        self.decoder2 =   nn.Conv2d(32,16, 3, stride=1, padding=1)
        self.de2_bn = nn.BatchNorm2d(16)
        self.decoder3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.de3_bn = nn.BatchNorm2d(8)

        self.decoderf1 =   nn.Conv2d(64, 32, 3, stride=1, padding=1)
        self.def1_bn = nn.BatchNorm2d(32)
        self.decoderf2=   nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.def2_bn = nn.BatchNorm2d(16)
        self.decoderf3 =   nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.def3_bn = nn.BatchNorm2d(8)

        self.encoderf1 =   nn.Conv2d(in_channels, 16, 3, stride=1, padding=1)  # First Layer GrayScale Image , change to input channels to 3 in case of RGB 
        self.enf1_bn = nn.BatchNorm2d(16)
        self.encoderf2=   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.enf2_bn = nn.BatchNorm2d(32)
        self.encoderf3 =   nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.enf3_bn = nn.BatchNorm2d(64)

        self.intere1_1 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.inte1_1bn = nn.BatchNorm2d(16)
        self.intere2_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.inte2_1bn = nn.BatchNorm2d(32)
        self.intere3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.inte3_1bn = nn.BatchNorm2d(64)

        self.intere1_2 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.inte1_2bn = nn.BatchNorm2d(16)
        self.intere2_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.inte2_2bn = nn.BatchNorm2d(32)
        self.intere3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.inte3_2bn = nn.BatchNorm2d(64)

        self.interd1_1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.intd1_1bn = nn.BatchNorm2d(32)
        self.interd2_1 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.intd2_1bn = nn.BatchNorm2d(16)
        self.interd3_1 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd3_1bn = nn.BatchNorm2d(64)

        self.interd1_2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        self.intd1_2bn = nn.BatchNorm2d(32)
        self.interd2_2 = nn.Conv2d(16,16,3, stride=1, padding=1)
        self.intd2_2bn = nn.BatchNorm2d(16)
        self.interd3_2 = nn.Conv2d(64,64,3, stride=1, padding=1)
        self.intd3_2bn = nn.BatchNorm2d(64)

        self.final = nn.Conv2d(8,n_classes,1,stride=1,padding=0)
        
        self.soft = nn.Softmax(dim =1)
    
    def forward(self, x):

        out = F.relu(self.en1_bn(F.max_pool2d(self.encoder1(x),2,2)))  #U-Net branch
        out1 = F.relu(self.enf1_bn(F.interpolate(self.encoderf1(x),scale_factor=(2,2),mode ='bilinear'))) #Ki-Net branch
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte1_1bn(self.intere1_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear')) #CRFB
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte1_2bn(self.intere1_2(tmp))),scale_factor=(4,4),mode ='bilinear')) #CRFB
        
        u1 = out  #skip conn
        o1 = out1  #skip conn

        out = F.relu(self.en2_bn(F.max_pool2d(self.encoder2(out),2,2)))
        out1 = F.relu(self.enf2_bn(F.interpolate(self.encoderf2(out1),scale_factor=(2,2),mode ='bilinear')))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte2_1bn(self.intere2_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte2_2bn(self.intere2_2(tmp))),scale_factor=(16,16),mode ='bilinear'))
        
        u2 = out
        o2 = out1

        out = F.relu(self.en3_bn(F.max_pool2d(self.encoder3(out),2,2)))
        out1 = F.relu(self.enf3_bn(F.interpolate(self.encoderf3(out1),scale_factor=(2,2),mode ='bilinear')))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.inte3_1bn(self.intere3_1(out1))),scale_factor=(0.015625,0.015625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.inte3_2bn(self.intere3_2(tmp))),scale_factor=(64,64),mode ='bilinear'))
        
        ### End of encoder block

        ### Start Decoder
        
        out = F.relu(self.de1_bn(F.interpolate(self.decoder1(out),scale_factor=(2,2),mode ='bilinear')))  #U-NET
        out1 = F.relu(self.def1_bn(F.max_pool2d(self.decoderf1(out1),2,2))) #Ki-NET
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intd1_1bn(self.interd1_1(out1))),scale_factor=(0.0625,0.0625),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd1_2bn(self.interd1_2(tmp))),scale_factor=(16,16),mode ='bilinear'))
        
        out = torch.add(out,u2)  #skip conn
        out1 = torch.add(out1,o2)  #skip conn

        out = F.relu(self.de2_bn(F.interpolate(self.decoder2(out),scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def2_bn(F.max_pool2d(self.decoderf2(out1),2,2)))
        tmp = out
        out = torch.add(out,F.interpolate(F.relu(self.intd2_1bn(self.interd2_1(out1))),scale_factor=(0.25,0.25),mode ='bilinear'))
        out1 = torch.add(out1,F.interpolate(F.relu(self.intd2_2bn(self.interd2_2(tmp))),scale_factor=(4,4),mode ='bilinear'))
        
        out = torch.add(out,u1)
        out1 = torch.add(out1,o1)

        out = F.relu(self.de3_bn(F.interpolate(self.decoder3(out),scale_factor=(2,2),mode ='bilinear')))
        out1 = F.relu(self.def3_bn(F.max_pool2d(self.decoderf3(out1),2,2)))

        

        out = torch.add(out,out1) # fusion of both branches

        out = F.relu(self.final(out))  #1*1 conv
        

#         out = self.soft(out)
        
        return out
    
    
    


""" Convolutional block:
    It follows a two 3x3 convolutional layer, each followed by a batch normalization and a relu activation.
"""
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
#         print(x.shape)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=3, stride=2, padding=1,output_padding=1)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        """ Encoder """
        self.e1 = encoder_block(1, 32)
        self.e2 = encoder_block(32, 64)
        self.e3 = encoder_block(64,128)
        self.e4 = encoder_block(128,256)

        """ Bottleneck """
        self.b = conv_block(256, 512)

        """ Decoder """
        self.d1 = decoder_block(512, 256)
        self.d2 = decoder_block(256,128)
        self.d3 = decoder_block(128,64)
        self.d4 = decoder_block(64,32)

        """ Classifier """
        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)

#         self.outputs = nn.Conv2d(8, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)
#         b = self.b(p2)

        """ Decoder """
#         d1 = self.d1(b, s2)
#         d2 = self.d2(d1, s1)
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Classifier """
#         outputs = self.outputs(d2)

        outputs = self.outputs(d4)
        
#         print(f's1,p1:{s1.shape,p1.shape}')
#         print(f's2,p2:{s2.shape,p2.shape}')
#         print(f's3,p3:{s3.shape,p3.shape}')
#         print(f's4,p4:{s4.shape,p4.shape}')
#         print(f'b:{b.shape}')
#         print(f'd1:{d1.shape}')
#         print(f'd2:{d2.shape}')
#         print(f'd3:{d3.shape}')
#         print(f'd4:{d4.shape}')
#         print(f'outputs:{outputs.shape}')

        
        return outputs
