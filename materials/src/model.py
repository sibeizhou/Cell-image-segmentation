import torch
import torch.nn as nn
import torch.nn.functional as F

#import any other libraries you need below this line
from torchvision.transforms.functional import center_crop as CenterCrop

class twoConvBlock(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(twoConvBlock, self).__init__()
    #initialize the block
    self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0)
    self.bn = nn.BatchNorm2d(out_channels)
    self.dropout = nn.Dropout(p=0.2)

  def forward(self, x):
    #implement the forward path
    x = self.relu(self.conv1(x))
    x = self.relu(self.bn(self.conv2(x)))
    x = self.dropout(x)
    return x

class downStep(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(downStep, self).__init__()
    #initialize the down path
    self.conv_block = twoConvBlock(in_channels, out_channels)
    self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    x = self.conv_block(x)
    x_down_sampled = self.max_pool(x)
    return x, x_down_sampled

class upStep(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(upStep, self).__init__()
    #initialize the up path
    self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_block = twoConvBlock(in_channels, out_channels)


  def forward(self, x, skip_connection):
    #implement the forward path
    x = self.up_conv(x)
    target_size = (x.size(2), x.size(3))
    skip_connection = CenterCrop(skip_connection, output_size=target_size)
    # skip_connection = self.crop_tensor(skip_connection, x)
    x = torch.cat([x, skip_connection], dim=1)
    x = self.conv_block(x)
    return x


class UNet(nn.Module):
  def __init__(self, in_channels=1, out_channels=2):
      super(UNet, self).__init__()
      self.down1 = downStep(in_channels, 64)
      self.down2 = downStep(64, 128)
      self.down3 = downStep(128, 256)
      self.down4 = downStep(256, 512)

      self.bottleneck = twoConvBlock(512, 1024)

      self.up1 = upStep(1024, 512)
      self.up2 = upStep(512, 256)
      self.up3 = upStep(256, 128)
      self.up4 = upStep(128, 64)

      self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

  def forward(self, x):
      x1, x_down1 = self.down1(x)
      x2, x_down2 = self.down2(x_down1)
      x3, x_down3 = self.down3(x_down2)
      x4, x_down4 = self.down4(x_down3)

      x_bottleneck = self.bottleneck(x_down4)

      x = self.up1(x_bottleneck, x4)
      x = self.up2(x, x3)
      x = self.up3(x, x2)
      x = self.up4(x, x1)

      # Final output layer (to match output channels)
      x = self.final_conv(x)

      return x
