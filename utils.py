import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import config
import os

transform = A.Compose([
            A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
            # A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(p=0.5),
            A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=255),
            ToTensorV2()
        ])
transform_eval = A.Compose([
            A.Resize(height=config.IMAGE_SIZE, width=config.IMAGE_SIZE),
            A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=255),
            ToTensorV2()
        ])
transform_original = A.Compose([
            A.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5], max_pixel_value=255),
            ToTensorV2()
        ])


def tensorboard_plotting(gen, imgs, cls, fakes, masks, flows, reconstruct_flows, writer, d_loss, g_loss, d_real_loss, d_fake_loss, d_real_cls, gp, cycle_loss, smooth_loss, g_fake_loss, mok_loss,tensorboard_step):
    gen.eval()
    with torch.no_grad():
        writer.add_scalar("0_d_loss", d_loss, global_step=tensorboard_step)
        writer.add_scalar("0_g_loss", g_loss, global_step=tensorboard_step)
        writer.add_scalar("d_real_loss", d_real_loss, global_step=tensorboard_step)
        writer.add_scalar("d_fake_loss", d_fake_loss, global_step=tensorboard_step)
        writer.add_scalar("d_real_cls", d_real_cls * config.LAMBDA_CLS, global_step=tensorboard_step)
        writer.add_scalar("gp", gp * config.LAMBDA_GP, global_step=tensorboard_step)
        writer.add_scalar("cycle_loss", cycle_loss * config.LAMBDA_CYCLE, global_step=tensorboard_step)
        writer.add_scalar("smooth_loss", smooth_loss * config.LAMBDA_SMOOTH, global_step=tensorboard_step)
        writer.add_scalar("g_fake_loss", g_fake_loss, global_step=tensorboard_step)
        writer.add_scalar("mok_loss", mok_loss * config.LAMBDA_MASK, global_step=tensorboard_step)
        if tensorboard_step % 10 == 0:
            # Plotting training result1
            batch_flow = torch.Tensor(flows_to_image(flows.detach().cpu())).permute(0, 3, 1, 2).to(config.DEVICE)
            batch_reconstruct_flow = torch.Tensor(flows_to_image(reconstruct_flows.detach().cpu())).permute(0, 3, 1, 2).to(config.DEVICE)
            visualize = torch.cat(
                [make_grid(imgs[:8] * 0.5 + 0.5),
                 make_grid(batch_flow[:8] * 0.5 + 0.5),
                 make_grid(fakes[:8] * 0.5 + 0.5),
                 make_grid(batch_reconstruct_flow[:8] * 0.5 + 0.5),
                 make_grid(masks[:8])], dim=1)
            writer.add_image("Training", visualize, global_step=tensorboard_step)

        #Plotting evaluation results on high resolution images(1024x1024)
        if tensorboard_step % 100 == 0:
            eval_imgs_list = os.listdir("example")
            eval_imgs = []
            eval_imgs_tranform = []
            for img in eval_imgs_list:
                img = Image.open(os.path.join("example", img))
                album = transform_eval(image=np.array(img))
                eval_imgs_tranform.append(album["image"].unsqueeze(0))
                album = transform_original(image=np.array(img))
                eval_imgs.append(album["image"].unsqueeze(0))


            eval_imgs = torch.cat(eval_imgs, dim=0).to(config.DEVICE)
            eval_imgs_tranform = torch.cat(eval_imgs_tranform, dim=0).to(config.DEVICE)

            upsample = nn.Upsample(scale_factor=8, mode="bilinear")
            upsampled_warp = dense_warp_field(shape=[1024, 1024], batch=eval_imgs.shape[0], device=config.DEVICE)


            cls = torch.Tensor([0, 1]).repeat(eval_imgs_tranform.shape[0], 1)
            eval_warp = upsample(gen(eval_imgs_tranform, cls.to(config.DEVICE)).permute(0, 3, 1, 2))
            eval_warp_visualize = torch.Tensor(flows_to_image(eval_warp.detach().cpu())).to(config.DEVICE)

            eval_fake_imgs = upsampled_warp(eval_imgs, eval_warp.permute(0, 2, 3, 1) / config.IMAGE_SIZE * 1024)
            eval_result = torch.cat([make_grid(eval_imgs * 0.5 + 0.5), make_grid(eval_fake_imgs * 0.5 + 0.5), make_grid(eval_warp_visualize.permute(0, 3, 1, 2) * 0.5 + 0.5)], dim = 1)
            save_image(eval_result, "result/"+str(tensorboard_step)+".png")
            # writer.add_image("eval_result", eval_result, global_step=tensorboard_step)

    gen.train()
    return tensorboard_step + 1


#Warping
# class dense_warp_field(nn.Module):
#     def __init__(self, shape=[config.IMAGE_SIZE,config.IMAGE_SIZE], batch = config.BATCH_SIZE, device = config.DEVICE):
#         super().__init__()
#         self.b = batch
#         self.h, self.w = shape
#
#         self.yy, self.xx = torch.meshgrid(torch.arange(0, self.w), torch.arange(0, self.h))
#         self.xx = self.xx.view(1, 1, self.h, self.w).repeat(self.b, 1, 1, 1)
#         self.yy = self.yy.view(1, 1, self.h, self.w).repeat(self.b, 1, 1, 1)
#         self.grid = torch.cat((self.xx, self.yy), 1).float() / self.w
#         self.grid = self.grid.permute(0,2,3,1).to(device)
#
#     def forward(self, imgs, flows, pad_mode = "zeros"):
#         # vgrid = Variable(self.grid, requires_grad=True) + flows
#         vgrid = self.grid + flows
#         # vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(self.w - 1, 1) - 1.0
#         # vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(self.h - 1, 1) - 1.0
#
#         output = nn.functional.grid_sample(imgs, vgrid, padding_mode=pad_mode)
#         return output

class dense_warp_field(nn.Module):
    def __init__(self, shape=[config.IMAGE_SIZE,config.IMAGE_SIZE], batch = config.BATCH_SIZE, device = config.DEVICE):
        super().__init__()
        self.b = batch
        self.h, self.w = shape

        self.yy, self.xx = torch.meshgrid(torch.arange(0, self.w), torch.arange(0, self.h))
        self.xx = self.xx.view(1, 1, self.h, self.w).repeat(self.b, 1, 1, 1)
        self.yy = self.yy.view(1, 1, self.h, self.w).repeat(self.b, 1, 1, 1)
        self.grid = torch.cat((self.xx, self.yy), 1).float()
        self.grid = self.grid.permute(0,2,3,1).to(device)

    def forward(self, imgs, flows, pad_mode = "zeros"):
        # vgrid = Variable(self.grid, requires_grad=True) + flows
        vgrid = self.grid + flows
        vgrid[:, :, :, 0] = 2.0 * vgrid[:, :, :, 0] / max(self.w - 1, 1) - 1.0
        vgrid[:, :, :, 1] = 2.0 * vgrid[:, :, :, 1] / max(self.h - 1, 1) - 1.0

        output = nn.functional.grid_sample(imgs, vgrid, padding_mode=pad_mode)
        return output

#loss
def gradient_panelty(discriminator, real, fake):
    B,C,H,W = real.shape
    epsilon = torch.rand((B, 1, 1, 1)).to(config.DEVICE)
    interpolated_images = epsilon*real + (1-epsilon)*fake.detach()
    interpolated_images.requires_grad_(True)
    interpolated_score, _ = discriminator(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=interpolated_score,
        grad_outputs=torch.ones_like(interpolated_score),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0],-1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_panelty = torch.mean((gradient_norm - 1)**2)
    return gradient_panelty

class warp_cycle_loss(nn.Module):
    def __init__(self, warpfield_shape = [config.IMAGE_SIZE,config.IMAGE_SIZE], batch = config.BATCH_SIZE, device = config.DEVICE):
        super().__init__()
        self.mse = nn.MSELoss()
        self.warp = dense_warp_field(device=device)
        self.h, self.w = warpfield_shape
        self.b = batch
        self.yy, self.xx = torch.meshgrid(torch.arange(0, self.w), torch.arange(0, self.h))
        self.xx = self.xx.view(1, 1, self.h, self.w).repeat(self.b, 1, 1, 1)
        self.yy = self.yy.view(1, 1, self.h, self.w).repeat(self.b, 1, 1, 1)
        self.A = torch.cat((self.xx, self.yy), 1).float().to(device)

    def forward(self, warpfield, warpfield_reconstruct):
        return self.mse(self.warp(self.warp(self.A, warpfield), warpfield_reconstruct), self.A)

class warp_smoothness_loss(nn.Module):
    def __init__(self, shape = [config.IMAGE_SIZE,config.IMAGE_SIZE]):
        super().__init__()
        self.mse = nn.MSELoss()
        self.n = (shape[0] - 1)*(shape[1] - 1)

    def forward(self, warpfield):
        return self.mse(warpfield[:, 1:, :-1, :], warpfield[:, :-1, :-1,:]) + self.mse(warpfield[:, :-1, 1:,:],warpfield[:,:-1,:-1,:]) / self.n

class warp_over_mask(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_logit = nn.BCEWithLogitsLoss()
    def forward(self, mask, flow):
        flow = flow.permute(0, 3, 1, 2)
        # return self.bce_logit(torch.sum(abs((1-mask))*abs(flow)), torch.sum(abs((mask))*abs(flow)))
        # return -(torch.sum(abs((mask))*abs(flow))-torch.sum(abs((1-mask))*abs(flow))) / mask.shape[2]*mask.shape[3]
        return torch.sum(abs((1-mask))*abs(flow))/torch.sum(abs(1-mask))

#Optical flow drawing
UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - \
                                  np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC,
    2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - \
                                  np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM,
    0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - \
                                  np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    assert len(flow.shape) == 3, "should not be batch here"

    if type(flow) == torch.Tensor:
        # check type
        flow = flow.numpy()

    if flow.shape[2] < 4:
        pass
    else:
        flow = flow.transpose(1, 2, 0)

    # import ipdb; ipdb.set_trace()
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)
    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def flows_to_image(flows):
    images = []
    if len(flows.shape) == 4:
        for i in range(flows.shape[0]):
            images.append(flow_to_image(flows[i]))
    else:
        images.append(flow_to_image(flows))

    return np.stack(images) / 255

# ————————————————
# credit: Evan_Tech
# Links: https://blog.csdn.net/yfren1123/article/details/104215553


#checkpoint(Credit: Alaadin persson)
def save_checkpoint(model, optimizer, epoch,filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    return epoch

if __name__ == "__main__":
    yy, xx = torch.meshgrid(torch.arange(0, 128), torch.arange(0,128))
    xx = xx.view(1, 1, 128, 128).repeat(3, 1, 1, 1)
    yy = yy.view(1, 1, 128, 128).repeat(3, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()
    grid = grid.permute(0, 2, 3, 1)
    print(grid[1,34,54])