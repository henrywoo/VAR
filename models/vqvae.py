"""
References:
- VectorQuantizer2: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L213
- GumbelQuantize: https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- VQVAE (VQModel): https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from basic_vae import Decoder, Encoder
from quant import VectorQuantizer2


class VQVAE(nn.Module):
    def __init__(
        self,
        vocab_size=4096,
        z_channels=32,
        ch=128,
        dropout=0.0,
        beta=0.25,  # commitment loss weight
        using_znorm=False,  # whether to normalize when computing the nearest neighbors
        quant_conv_ks=3,  # quant conv kernel size
        quant_resi=0.5,  # 0.5 means \phi(x) = 0.5conv(x) + (1-0.5)x
        share_quant_resi=4,  # use 4 \phi layers for K scales: partially-shared \phi
        default_qresi_counts=0,  # if is 0: automatically set to len(v_patch_nums)
        v_patch_nums=(
            1,
            2,
            3,
            4,
            5,
            6,
            8,
            10,
            13,
            16,
        ),  # number of patches for each scale, h_{1 to K} = w_{1 to K} = v_patch_nums[k]
        test_mode=True,
    ):
        super().__init__()
        self.test_mode = test_mode
        self.V, self.Cvae = vocab_size, z_channels
        # ddconfig is copied from https://github.com/CompVis/latent-diffusion/blob/e66308c7f2e64cb581c6d27ab6fbeb846828253b/models/first_stage_models/vq-f16/config.yaml
        ddconfig = dict(
            dropout=dropout,
            ch=ch,
            z_channels=z_channels,
            in_channels=3,
            ch_mult=(1, 1, 2, 2, 4),
            num_res_blocks=2,  # from vq-f16/config.yaml above
            using_sa=True,
            using_mid_sa=True,  # from vq-f16/config.yaml above
            # resamp_with_conv=True,   # always True, removed.
        )
        ddconfig.pop("double_z", None)  # only KL-VAE should use double_z=True
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig["ch_mult"]) - 1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size,
            Cvae=self.Cvae,
            using_znorm=using_znorm,
            beta=beta,
            default_qresi_counts=default_qresi_counts,
            v_patch_nums=v_patch_nums,
            quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(
            self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2
        )
        self.post_quant_conv = torch.nn.Conv2d(
            self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2
        )

        if self.test_mode:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]

    # ===================== `forward` is only used in VAE training =====================
    def forward(self, inp, ret_usages=False):  # -> rec_B3HW, idx_N, loss
        # VectorQuantizer2.forward
        t0 = self.encoder(inp)
        f_hat, usages, vq_loss = self.quantize(self.quant_conv(t0), ret_usages=ret_usages)
        t1 = self.post_quant_conv(f_hat)
        t2 = self.decoder(t1)
        return t2, usages, vq_loss

    # ===================== `forward` is only used in VAE training =====================

    def fhat_to_img(self, f_hat: torch.Tensor):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)

    def img_to_idxBl(
        self,
        inp_img_no_grad: torch.Tensor,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
    ) -> List[torch.LongTensor]:  # return List[Bl]
        f = self.quant_conv(self.encoder(inp_img_no_grad))
        return self.quantize.f_to_idxBl_or_fhat(
            f, to_fhat=False, v_patch_nums=v_patch_nums
        )

    def idxBl_to_img(
        self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BChw = []
        for idx_Bl in ms_idx_Bl:
            l = idx_Bl.shape[1]
            pn = round(l**0.5)
            ms_h_BChw.append(
                self.quantize.embedding(idx_Bl)
                .transpose(1, 2)
                .view(B, self.Cvae, pn, pn)
            )
        return self.embed_to_img(
            ms_h_BChw=ms_h_BChw, all_to_max_scale=same_shape, last_one=last_one
        )

    def embed_to_img(
        self, ms_h_BChw: List[torch.Tensor], all_to_max_scale: bool, last_one=False
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        if last_one:
            return self.decoder(
                self.post_quant_conv(
                    self.quantize.embed_to_fhat(
                        ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=True
                    )
                )
            ).clamp_(-1, 1)
        else:
            return [
                self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
                for f_hat in self.quantize.embed_to_fhat(
                    ms_h_BChw, all_to_max_scale=all_to_max_scale, last_one=False
                )
            ]

    def img_to_reconstructed_img(
        self,
        x,
        v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None,
        last_one=False,
    ) -> List[torch.Tensor]:
        f = self.quant_conv(self.encoder(x))
        ls_f_hat_BChw = self.quantize.f_to_idxBl_or_fhat(
            f, to_fhat=True, v_patch_nums=v_patch_nums
        )
        if last_one:
            return self.decoder(self.post_quant_conv(ls_f_hat_BChw[-1])).clamp_(-1, 1)
        else:
            return [
                self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
                for f_hat in ls_f_hat_BChw
            ]

    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if (
            "quantize.ema_vocab_hit_SV" in state_dict
            and state_dict["quantize.ema_vocab_hit_SV"].shape[0]
            != self.quantize.ema_vocab_hit_SV.shape[0]
        ):
            state_dict["quantize.ema_vocab_hit_SV"] = self.quantize.ema_vocab_hit_SV
        return super().load_state_dict(
            state_dict=state_dict, strict=strict, assign=assign
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch
    from hiq import print_model
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchvision import transforms
    from hiq.cv_torch import get_cv_dataset, DS_PATH_IMAGENET1K
    from tqdm import tqdm


    # 定义去归一化变换
    def denormalize(tensor, mean, std):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor


    # 数据预处理和数据加载
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    loader_params = dict(
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    dataloader = get_cv_dataset(path=DS_PATH_IMAGENET1K,
                                image_size=256,
                                split='validation',
                                batch_size=50,
                                num_workers=5,
                                transform=transform,
                                return_type="pair",
                                return_loader=True,
                                convert_rgb=True,
                                **loader_params
                                )

    # 加载预训练的权重文件
    pretrained_weights_path = 'vae_ch160v4096z32.pth'

    # 实例化 VQVAE 模型，使用与预训练模型匹配的参数
    vqvae = VQVAE(ch=160)

    # 加载预训练权重，设置 strict=True
    vqvae.load_state_dict(torch.load(pretrained_weights_path), strict=True)

    # 将模型移动到 GPU，如果可用的话
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqvae.to(device)
    vqvae.eval()
    print_model(vqvae)

    # 准备 FID 计算器
    fid = FrechetInceptionDistance(feature=2048)

    # 计算 FID 分数
    transform_fid = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    # 遍历 dataloader 中的所有批次
    for batch, _ in tqdm(dataloader, desc="Processing batches", colour="09ff23"):
        batch = batch.to(device)

        # 前向传递
        with torch.no_grad():
            reconstructed_batch, _, _ = vqvae(batch)

        # 去归一化
        batch = denormalize(batch, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        reconstructed_batch = denormalize(reconstructed_batch, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        # 将值限制在 [0, 1] 范围内，然后转换为 uint8 类型
        batch = torch.clamp(batch, 0, 1) * 255
        reconstructed_batch = torch.clamp(reconstructed_batch, 0, 1) * 255

        # 将原始和重建图像转换为 uint8 类型
        batch_uint8 = batch.byte().cpu().numpy().transpose(0, 2, 3, 1)
        reconstructed_batch_uint8 = reconstructed_batch.byte().cpu().numpy().transpose(0, 2, 3, 1)

        # 计算并更新 FID 分数
        for i in range(batch.size(0)):
            original_image_fid = transform_fid(batch_uint8[i]).unsqueeze(0)
            reconstructed_image_fid = transform_fid(reconstructed_batch_uint8[i]).unsqueeze(0)
            fid.update(original_image_fid.type(torch.uint8), real=True)
            fid.update(reconstructed_image_fid.type(torch.uint8), real=False)

    fid_score = fid.compute()

    print(f"Reconstruction FID score: {fid_score:.2f}")

    # 显示一个原始图像和对应的重建图像示例
    original_image_example = batch_uint8[0]
    reconstructed_image_example = reconstructed_batch_uint8[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original_image_example)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(reconstructed_image_example)
    axes[1].set_title(f"Reconstructed Image\nFID: {fid_score:.2f}")
    axes[1].axis("off")
    plt.savefig("astronaut.png")
    plt.show()

'''
- Imagenette:
Reconstruction FID score: 0.76
'''