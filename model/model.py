# MODEL ARCHITECTURE
 # 3x3 convolution with padding
def conv3x3(in_planes,out_planes,stride=1,dilation=1):  
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,
                     stride=stride,padding=dilation,
                     bias=False,dilation=dilation,)
 # 1x1 convolution
def conv1x1(in_planes,out_planes,stride=1):                     
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
    
class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self,in_planes,planes,stride=1,dilation=1):  
        super().__init__()
        self.norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(in_planes,planes)
        self.bn1 = self.norm_layer(planes)
        self.conv2 = conv3x3(planes,planes,stride,dilation)
        self.bn2 = self.norm_layer(planes)
        self.conv3 = conv1x1(planes,self.expansion*planes)
        self.bn3 = self.norm_layer(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        if self.stride!=1 or in_planes!=planes*self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes,planes*self.expansion,kernel_size=1,stride=self.stride,bias=False),
                nn.BatchNorm2d(planes*self.expansion),
            )
        else:
            self.downsample = None
    def forward(self,x):
        x_skip = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        if self.downsample is not None:
            x_skip = self.downsample(x_skip)
        x += x_skip
        x = self.relu(x)
        return x

class ResNet50(nn.Module):
    def __init__(self,num_classes=1000):
        super().__init__()
        self.layers = [3,4,6,3]   # number of blocks in each layer
        self.expansion = 4
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3,self.in_planes,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer1 = self._make_layer(Bottleneck,64,self.layers[0])
        self.layer2 = self._make_layer(Bottleneck,128,self.layers[1],stride=2)
        self.layer3 = self._make_layer(Bottleneck,256,self.layers[2],stride=2)
        self.layer4 = self._make_layer(Bottleneck,512,self.layers[3],stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion,num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,block,planes,blocks,stride=1,dilation=1):
        layers=[]
        layers.append(block(self.in_planes,planes,stride,dilation))
        self.in_planes = planes * block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.in_planes,planes))
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
