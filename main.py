import torch, config, shutil, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from model import Generator,Discriminator
from Dataset import RealPastedGlasses
from tqdm import tqdm
from utils import *

def get_d_loss(gen, critic, img, cls, target_cls, warp, CELoss):
    fake_warp = gen(img, target_cls)
    fake = warp(img, fake_warp)
    pred_score, pred_cls = critic(img)
    d_real_loss = torch.mean(pred_score)
    d_real_cls = CELoss(pred_cls, torch.argmax(cls, dim=1))

    pred_score, pred_cls = critic(fake.detach())
    d_fake_loss = torch.mean(pred_score)

    gp = gradient_panelty(critic, img, fake)
    d_loss = -(d_real_loss - d_fake_loss) + config.LAMBDA_CLS * d_real_cls + config.LAMBDA_GP * gp
    return d_loss, fake, fake_warp, d_real_loss.item(), d_fake_loss.item(), d_real_cls.item(), gp.item()

def get_g_loss(gen, critic, cls, target_cls, fake, fake_warp, CELoss, cycle, smooth, mok, masks):
    pred_score, pred_cls = critic(fake)
    g_fake_loss = torch.mean(pred_score)
    g_fake_cls = CELoss(pred_cls, torch.argmax(target_cls, dim=1))

    warp_reconstruct = gen(fake, cls)
    cycle_loss = cycle(fake_warp, warp_reconstruct)
    smooth_loss = smooth(fake_warp) + smooth(warp_reconstruct)
    mok_loss = mok(masks, fake_warp) + mok(masks, warp_reconstruct)

    g_loss = -g_fake_loss + config.LAMBDA_CLS * g_fake_cls + config.LAMBDA_CYCLE * cycle_loss + config.LAMBDA_SMOOTH * smooth_loss + config.LAMBDA_MASK * mok_loss
    return g_loss, warp_reconstruct, cycle_loss.item(), smooth_loss.item(), g_fake_loss.item(), mok_loss.item()

def train(gen, critic, optim_g, optim_d, g_scaler, d_scaler, epoch, Dataloader, writer, tensorboard_step):
    loop = tqdm(Dataloader, leave=True, bar_format='{l_bar}{bar:60}{r_bar}{bar:-10b}')
    CELoss = nn.CrossEntropyLoss()
    warp = dense_warp_field(device=config.DEVICE)
    cycle = warp_cycle_loss(device=config.DEVICE)
    smooth = warp_smoothness_loss()
    mok = warp_over_mask()
    g_loss = 0
    for idx, (imgs, masks, cls) in enumerate(loop):
        imgs = imgs.to(config.DEVICE)
        masks = masks.to(config.DEVICE)
        cls = cls.to(config.DEVICE)
        target_cls = torch.abs(cls - 1.0).float() # Transfer to another target domain

        #Mix precision training
        if config.FLOAT16:
            with torch.cuda.amp.autocast():
                d_loss, fakes, fake_warps, d_real_loss, d_fake_loss, d_real_cls, gp = get_d_loss(gen, critic, imgs, cls, target_cls, warp, CELoss)
            optim_d.zero_grad()
            d_scaler.scale(d_loss).backward()
            d_scaler.step(optim_d)
            d_scaler.update()

            if (idx + 1) % config.CRITIC == 0:
                with torch.cuda.amp.autocast():
                    g_loss, warp_reconstructs, cycle_loss, smooth_loss, g_fake_loss, mok_loss = get_g_loss(gen, critic, cls, target_cls,fakes, fake_warps, CELoss, cycle, smooth, mok, masks)
                optim_g.zero_grad()
                g_scaler.scale(g_loss).backward()
                g_scaler.step(optim_g)
                g_scaler.update()
                tensorboard_step = tensorboard_plotting(gen, imgs, cls, fakes, masks, fake_warps, warp_reconstructs, writer,
                                                        d_loss, g_loss, d_real_loss, d_fake_loss, d_real_cls, gp, cycle_loss, smooth_loss, g_fake_loss, mok_loss,tensorboard_step)

        #Normal training
        else:
            d_loss, fakes, fake_warps, d_real_loss, d_fake_loss, d_real_cls, gp = get_d_loss(gen, critic, imgs, cls, target_cls, warp, CELoss)
            optim_d.zero_grad()
            d_loss.backward()
            optim_d.step()


            if (idx + 1) % config.CRITIC == 0:
                g_loss, warp_reconstructs, cycle_loss, smooth_loss, g_fake_loss, mok_loss = get_g_loss(gen, critic, cls, target_cls, fakes, fake_warps, CELoss, cycle, smooth, mok, masks)
                optim_g.zero_grad()
                g_loss.backward()
                optim_g.step()
                tensorboard_step = tensorboard_plotting(gen, imgs, cls, fakes, masks, fake_warps, warp_reconstructs, writer,
                                                        d_loss, g_loss, d_real_loss, d_fake_loss, d_real_cls, gp, cycle_loss, smooth_loss, g_fake_loss, mok_loss,tensorboard_step)

        g_loss_postfix = g_loss if isinstance(g_loss, int) else g_loss.item()
        loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss_postfix, max_warping_vector=torch.max(fake_warps).item(), min_warping_vector=torch.min(fake_warps).item())
        loop.set_description(f"Epoch:{epoch + 1}/{config.EPOCH}")
    return tensorboard_step


if __name__ == '__main__':
    #initia models, optimizers, and scalers
    gen = Generator(img_size=config.IMAGE_SIZE).to(config.DEVICE)
    critic = Discriminator(img_size=config.IMAGE_SIZE).to(config.DEVICE)
    optim_g = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    optim_d = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    #Weighted random sampling
    class_weight = [1 / len(os.listdir(config.ACTUAL_GLASSES_ROOT)), 1 / len(os.listdir(config.PASTED_GLASSES_ROOT))]
    real_sampleweight = [1 / len(os.listdir(config.ACTUAL_GLASSES_ROOT))] * len(os.listdir(config.ACTUAL_GLASSES_ROOT))
    pasted_sampleweight = [1 / len(os.listdir(config.PASTED_GLASSES_ROOT))] * len(os.listdir(config.PASTED_GLASSES_ROOT))
    sample_weight = real_sampleweight + pasted_sampleweight
    sampler = WeightedRandomSampler(sample_weight, num_samples=len(sample_weight), replacement=True)

    dataset = RealPastedGlasses(config.ACTUAL_GLASSES_ROOT, config.PASTED_GLASSES_ROOT, transform=transform)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, pin_memory=True, drop_last=True, num_workers=config.NUM_WORKERS, sampler=sampler)

    shutil.rmtree(f"E:/cache")
    os.makedirs(f"E:/cache")
    writer = SummaryWriter(f"E:/cache")
    tensorboard_step = 0

    if config.LOAD_MODEL:
        load_checkpoint("g_checkpoint.pth.tar", gen, optim_g, config.LEARNING_RATE)
        load_checkpoint("d_checkpoint.pth.tar", critic, optim_d, config.LEARNING_RATE)
    for epoch in range(config.EPOCH):
        tensorboard_step = train(gen, critic, optim_g, optim_d, g_scaler, d_scaler, epoch, loader, writer, tensorboard_step)

        if config.SAVE_MODEL:
            save_checkpoint(gen, optim_g, epoch, "g_checkpoint.pth.tar")
            save_checkpoint(critic, optim_d, epoch, "d_checkpoint.pth.tar")

