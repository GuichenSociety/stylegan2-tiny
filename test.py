import config
from model import Discriminator, Generator, MappingNetwork
import math
import torch
from torchvision.utils import save_image


class test():
    def __init__(self,num_img):

        log_resolution = int(math.log2(config.Image_size))
        self.device = config.Device

        self.generator = Generator(log_resolution, config.D_latent).to(self.device)
        self.mapping_network = MappingNetwork(config.D_latent, config.Mapping_network_layers).to(self.device)

        self.n_gen_blocks = self.generator.n_blocks

        print("Load weights>>>")
        self.generator.load_state_dict(
            torch.load(f"weight/gen_{config.Image_size}x{config.Image_size}.pt", map_location=self.device))

        self.mapping_network.load_state_dict(
            torch.load(f"weight/map_{config.Image_size}x{config.Image_size}.pt", map_location=self.device))

        with torch.no_grad():
            folder = "result_img"

            for i in range(num_img):
                generated_images, w = self.generate_images(1)
                save_image(generated_images * 0.5 + 0.5, folder + f"/{i+1}.png")


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


if __name__ == '__main__':
    test(num_img=9)