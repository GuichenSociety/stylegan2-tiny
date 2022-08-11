import math
from pathlib import Path
from typing import Iterator, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision
from PIL import Image
from tqdm import tqdm

import config
from model import Discriminator, Generator, MappingNetwork, GradientPenalty, PathLengthPenalty
from utils import DiscriminatorLoss,GeneratorLoss

torch.backends.cudnn.benchmark = True

class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, image_size):
        super().__init__()
        self.paths = [p for p in Path(path).glob(f'**/*.{config.Image_format}')]
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)


    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)





class main():
    def __init__(self):
        self.device = config.Device
        self.batch_size = config.Batch_size

        dataset = Dataset(config.Dataset_path, config.Image_size)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=config.Num_workers,shuffle = True, drop_last = True, pin_memory = True)

        log_resolution = int(math.log2(config.Image_size))

        self.discriminator = Discriminator(log_resolution).to(self.device)
        self.generator = Generator(log_resolution, config.D_latent).to(self.device)
        self.mapping_network = MappingNetwork(config.D_latent, config.Mapping_network_layers).to(self.device)

        if config.Load_weights:
            print("Load weights>>>")
            self.generator.load_state_dict(torch.load(f"weight/gen_{config.Image_size}x{config.Image_size}.pt", map_location=self.device))
            self.discriminator.load_state_dict(torch.load(f"weight/dis_{config.Image_size}x{config.Image_size}.pt", map_location=self.device))
            self.mapping_network.load_state_dict(torch.load(f"weight/map_{config.Image_size}x{config.Image_size}.pt", map_location=self.device))

        self.n_gen_blocks = self.generator.n_blocks

        self.gradient_penalty = GradientPenalty()
        self.path_length_penalty = PathLengthPenalty(0.99).to(self.device)


        self.discriminator_loss = DiscriminatorLoss().to(self.device)
        self.generator_loss = GeneratorLoss().to(self.device)

        self.discriminator_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.Learning_rate,
            betas=config.Adam_betas
        )
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.Learning_rate,
            betas=config.Adam_betas
        )
        self.mapping_network_optimizer = torch.optim.Adam(
            self.mapping_network.parameters(),
            lr=config.Mapping_network_learning_rate, betas=config.Adam_betas
        )


    def get_w(self,batch_size):

        # 是否进行混合风格
        if torch.rand(()).item() < config.Style_mixing_prob:

            cross_over_point = int(torch.rand(()).item() * self.n_gen_blocks)
            z2 = torch.randn(batch_size, config.D_latent).to(self.device)
            z1 = torch.randn(batch_size, config.D_latent).to(self.device)

            w1 = self.mapping_network(z1)
            w2 = self.mapping_network(z2)

            w1 = w1[None, :, :].expand(cross_over_point, -1, -1)
            w2 = w2[None, :, :].expand(self.n_gen_blocks - cross_over_point, -1, -1)

            return torch.cat((w1, w2), dim=0)     # shape is (n_gen_blocks,batch_size,D_latent)

        else:

            z = torch.randn(batch_size, config.D_latent).to(self.device)

            w = self.mapping_network(z)

            return w[None, :, :].expand(self.n_gen_blocks, -1, -1)    # shape is (n_gen_blocks,batch_size,D_latent)

    def get_noise(self, batch_size):

        noise = []
        resolution = 4
        for i in range(self.n_gen_blocks):
            if i == 0:
                n1 = None
            else:
                n1 = torch.randn(batch_size, 1, resolution, resolution).to(self.device)

            n2 = torch.randn(batch_size, 1, resolution, resolution).to(self.device)
            noise.append((n1, n2))
            resolution *= 2

        # 获得噪声，noise[0]=(None,shape:(batch_size,1,4,4))  , noise[1] = (shape:(batch_size,1,8,8),shape:(batch_size,1,8,8)) ,...
        return noise

    def generate_images(self, batch_size):

        w = self.get_w(batch_size)
        noise = self.get_noise(batch_size)
        images = self.generator(w, noise)

        return images, w

    def save_checkpoint(self,model, filename="my_checkpoint.pt"):
        # print("=> Saving checkpoint")
        checkpoint = model.state_dict()
        torch.save(checkpoint, filename)

    def step(self, idx):
        loop = tqdm(self.dataloader, leave=True)
        for batch_idx, real in enumerate(loop):
            real_images = real.to(self.device)

            '''训练判别器'''
            self.discriminator_optimizer.zero_grad()


            for i in range(config.Gradient_accumulate_steps):


                generated_images, _ = self.generate_images(self.batch_size)
                fake_output = self.discriminator(generated_images.detach())

                if (batch_idx + 1) % config.Lazy_gradient_penalty_interval == 0:
                    real_images.requires_grad_()

                real_output = self.discriminator(real_images)

                real_loss, fake_loss = self.discriminator_loss(real_output, fake_output)
                disc_loss = real_loss + fake_loss

                if (batch_idx + 1) % config.Lazy_gradient_penalty_interval == 0:
                    gp = self.gradient_penalty(real_images, real_output)
                    disc_loss = disc_loss + 0.5 * config.Gradient_penalty_coefficient * gp * config.Lazy_gradient_penalty_interval


                disc_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)

            self.discriminator_optimizer.step()



            '''训练生成器，和潜在空间'''
            self.generator_optimizer.zero_grad()
            self.mapping_network_optimizer.zero_grad()

            for i in range(config.Gradient_accumulate_steps):


                generated_images, w = self.generate_images(self.batch_size)
                fake_output = self.discriminator(generated_images)
                gen_loss = self.generator_loss(fake_output)

                if batch_idx > config.Lazy_path_penalty_after and (batch_idx + 1) % config.Lazy_gradient_penalty_interval == 0:
                    plp = self.path_length_penalty(w, generated_images)

                    if not torch.isnan(plp):
                        gen_loss = gen_loss + plp


                gen_loss.backward()


            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.mapping_network.parameters(), max_norm=1.0)

            self.generator_optimizer.step()
            self.mapping_network_optimizer.step()



            if (batch_idx + 1) % config.Save_checkpoint_interval == 0:
                self.save_checkpoint(self.generator, filename=f"weight/gen_{config.Image_size}x{config.Image_size}.pt")
                self.save_checkpoint(self.discriminator, filename=f"weight/dis_{config.Image_size}x{config.Image_size}.pt")
                self.save_checkpoint(self.mapping_network, filename=f"weight/map_{config.Image_size}x{config.Image_size}.pt")

                with torch.no_grad():
                    folder = "gen_Image"
                    generated_images, w = self.generate_images(self.batch_size)
                    save_image(generated_images * 0.5 + 0.5, folder + f"/gen_{idx}_{batch_idx}.png")


if __name__ == '__main__':
    train = main()
    for i in range(config.Epochs):
        train.step(i)
