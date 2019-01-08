import torch
import torch.nn as nn
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import torch.nn.functional as F
import os


def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=has_bias)


def conv1x1_sigmoid(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv1x1(in_planes, out_planes, stride),
            nn.Sigmoid())


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv1x1(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)


class DeblurModel(nn.Module):

    def __init__(self, backbone_name):
        super(DeblurModel, self).__init__()
        self.backbone = eval(backbone_name)(pretrained=True) 
        if backbone_name in ["resnet18", "resnet34"]:
            expansion = 1
        elif backbone_name in ["resnet50", "resnet101", "resnet152"]:
            expansion = 4
        else:
            raise ValueError("Not support such backbone")
        self.deconv5 = nn.Sequential(
                conv1x1_bn_relu(512 * expansion, 256 * expansion),
                conv3x3_bn_relu(256 * expansion, 256 * expansion))
        self.deconv4 = nn.Sequential(
                conv1x1_bn_relu(256 * expansion, 128 * expansion),
                conv3x3_bn_relu(128 * expansion, 128 * expansion))
        self.deconv3 = nn.Sequential(
                conv1x1_bn_relu(128 * expansion, 64 * expansion),
                conv3x3_bn_relu(64 * expansion, 64 * expansion))
        self.deconv2 = nn.Sequential(
                conv1x1_bn_relu(64 * expansion, 64 * expansion),
                conv3x3_bn_relu(64 * expansion, 64 * expansion))
        self.pss_map = nn.Sequential(
                conv3x3_bn_relu(64 * expansion + 3, 64 * expansion),
                conv1x1(64 * expansion, 3))

        self.deconv4.apply(weights_init)
        self.deconv3.apply(weights_init)
        self.deconv2.apply(weights_init)

    def forward(self, x):
        c2, c3, c4, c5 = self.backbone(x)
        out4 = F.interpolate(self.deconv5(c5), size=(c4.size(2), c4.size(3))) + c4
        out3 = F.interpolate(self.deconv4(out4), size=(c3.size(2), c3.size(3))) + c3
        out2 = F.interpolate(self.deconv3(out3), size=(c2.size(2), c2.size(3))) + c2
        out1 = F.interpolate(self.deconv2(out2), size=(x.size(2), x.size(3)))
        out1 = torch.cat((out1, x), 1)
        pss_map = self.pss_map(out1)
        return pss_map


def save_state(ckpt_dir, epoch, network, optimizer):
    save_state_dict = {
            "network": network.state_dict(),
            "optimizer": optimizer.state_dict()}
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    save_filepath = os.path.join(ckpt_dir, "model-%d.pth" % epoch)
    torch.save(save_state_dict, save_filepath)
    print("=> saving checkpoint at %s." % save_filepath)


def load_state(ckpt_dir, epoch, network, optimizer=None):
    save_filepath = os.path.join(ckpt_dir, "model-%d.pth" % epoch)
    if os.path.isfile(save_filepath):
        checkpoint = torch.load(save_filepath)
        network.load_state_dict(checkpoint["network"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        print("=> restore checkpoint from %s finished." % save_filepath)
    else:
        print("=> no checkpoint found at %s." % save_filepath)


if __name__ == "__main__":
    x = torch.randn(2, 3, 300, 300)
    network = DeblurModel("resnet18")
    y = network(x)
    print(y.shape)

