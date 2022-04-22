import torch
import torch.nn as nn

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,down = True, use_act=True, use_bn = True):
        super(ConvModule, self).__init__()
        layers = []
        layers.append(nn.Upsample(scale_factor=2, mode="bilinear")) if not down else None
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False if use_bn else True))
        layers.append(nn.BatchNorm2d(out_channels) if use_bn else nn.InstanceNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True)) if use_act else None
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)


class ResidualModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.module = nn.Sequential(
            ConvModule(channels, channels, kernel_size = 3, stride = 1,padding = 1),
            ConvModule(channels, channels, use_act = False, stride = 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.module(x)


class Generator(nn.Module):
    def __init__(self, in_channels=3, img_size=144, num_classes=2, num_filters=[64, 128], num_residual=6):
        super().__init__()
        self.img_size = img_size
        self.init = nn.Sequential(
            nn.Conv2d(in_channels + num_classes, num_filters[0], 7, 1, 3, bias=False),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(inplace=True)
        )

        down = []
        for filter in num_filters:
            down.append(ConvModule(filter, filter * 2, down=True, kernel_size=4, stride=2, padding=1))
        self.down = nn.Sequential(*down)

        self.residual_bottleneck = nn.Sequential(
            *[ResidualModule(num_filters[-1] * 2) for _ in range(num_residual)]
        )

        up = []
        for filter in num_filters[::-1]:
            up.append(
                ConvModule(filter * 2, filter, down=False, kernel_size=4, stride=1, padding="same")
            )
        self.up = nn.Sequential(*up)
        self.decode = nn.Sequential(
            nn.Conv2d(num_filters[0], 2, 7, 1, 3, bias=False)
        )

    def forward(self, x, cls):
        cls = cls.view(cls.size(0), cls.size(1), 1, 1)
        cls = cls.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, cls], dim=1)
        x = self.init(x)
        x = self.down(x)
        x = self.residual_bottleneck(x)
        x = self.up(x)
        return torch.permute(self.decode(x), (0, 2, 3, 1))


class Discriminator(nn.Module):
    def __init__(self, in_channels = 3, img_size = 144,depth = 5, num_classes = 2, num_filters=64, padding_mode = "zeros"):
        super(Discriminator, self).__init__()
        self.init = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode),
            nn.LeakyReLU(0.01)
        )

        mainbody = []
        for i in range(1, depth):
            mainbody.append(nn.Conv2d(num_filters, num_filters*2, kernel_size=4, stride=2, padding=1, padding_mode=padding_mode)),
            mainbody.append(nn.LeakyReLU(0.01))
            num_filters*=2

        kernel_size = int(img_size/(2**depth))
        self.main = nn.Sequential(*mainbody)
        self.score = nn.Conv2d(num_filters, 1, kernel_size=3, stride=1, padding=1, padding_mode=padding_mode, bias=False)

        self.cls = nn.Sequential(
            nn.Conv2d(num_filters, num_classes, kernel_size=kernel_size, bias=False),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.init(x)
        x = self.main(x)
        return self.score(x),self.cls(x)



if __name__ == "__main__":
    from torchsummary import summary
    images = torch.rand(16, 3, 128, 128)
    cls = torch.zeros(16,2)
    model = Generator()
    output = model(images, cls)
    print(output.shape)
    outputimage = nn.functional.grid_sample(images, output)
    print(outputimage.shape)
    summary(model,[(3,128,128),(2,)],device="cpu")

    model = Discriminator()
    summary(model, (3, 128, 128), device="cpu")
