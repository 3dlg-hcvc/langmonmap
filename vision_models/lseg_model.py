# numpy
import math
import numpy as np

# modeling
from vision_models.base_model import BaseModel
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T
from torch.nn import functional as F
from fvcore.common.checkpoint import Checkpointer
import open_clip
import torch
from lseg.clip_mapping_utils import (
    get_new_pallete,
    get_new_mask_pallete,
)
from lseg.modules.models.lseg_net import LSegEncNet
from lseg.additional_utils.models import (
    resize_image,
    pad_image,
    crop_image,
)
import torchvision.transforms as TS

# Visualization
import matplotlib.pyplot as plt

# typing
from typing import List, Optional

from PIL import Image

try:
    import tensorrt as trt
except:
    print("TensorRT not available, cannot use Jetson")

class LSegModel(torch.nn.Module, BaseModel):
    def __init__(self,
                 jetson: bool = False,
                 fuse_similarity: str = "sum",
                 ):
        super(LSegModel, self).__init__()
        self.jetson = jetson
        self.fuse_similarity = fuse_similarity

        self.input_format = "RGB"

        self.aug = T.ResizeShortestEdge(
            [640, 640], 2560
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.clip_resolution = (512, 512)

        name, pretrain = ('ViT-B-32', 'laion2b_s34b_b79k')
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            name,
            pretrained=pretrain,
            device="cuda", )
        self.clip_model = clip_model.float()
        self.clip_model.eval()
        self.clip_preprocess = clip_preprocess
        self.clip_mean = torch.Tensor([122.7709383, 116.7460125, 104.09373615]).to("cuda")
        self.clip_mean = self.clip_mean.unsqueeze(-1).unsqueeze(-1)
        self.clip_std = torch.Tensor([68.5005327, 66.6321579, 70.3231630]).to("cuda")
        self.clip_std = self.clip_std.unsqueeze(-1).unsqueeze(-1)
        
        self.device="cuda"
        self.crop_size = 480  # 480
        self.base_size = 520  # 520
        self.lseg_model = LSegEncNet(
            "",
            arch_option=0,
            block_depth=0,
            activation="lrelu",
            crop_size=self.crop_size,
        )
        model_state_dict = self.lseg_model.state_dict()
        checkpoint_path = "weights/lseg.ckpt"
        pretrained_state_dict = torch.load(checkpoint_path, map_location=self.device)
        pretrained_state_dict = {
            k.lstrip("net."): v for k, v in pretrained_state_dict["state_dict"].items()
        }
        model_state_dict.update(pretrained_state_dict)
        self.lseg_model.load_state_dict(pretrained_state_dict)

        self.lseg_model.eval()
        self.lseg_model = self.lseg_model.to(self.device)

        self.norm_mean = [0.5, 0.5, 0.5]
        self.norm_std = [0.5, 0.5, 0.5]
        self.lseg_transform = TS.Compose(
            [
                TS.ToTensor(),
                TS.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.feature_dim = self.lseg_model.out_c
        self.labels = []

    def set_classes(self,
                    classes: List[str]
                    ):
        self.labels = classes

    def eval(self):
        super().eval()
        if self.jetson:
            pass
        else:
            self.clip_model.eval()
        return self

    def compute_similarity(self,
                           image_feats: torch.Tensor,
                           text_feats: torch.Tensor,
                           num_layers: float = 0,
                           ) -> torch.Tensor:
        # image_feats = F.normalize(image_feats, dim=1)  # B C H W, normalize along C
        # text_feats = F.normalize(text_feats, dim=1)
        # print(image_feats.max())
        # print(text_feats.max())
        if self.fuse_similarity == "layerwisemax":
            filter_height = text_feats.shape[0]
            if len(image_feats.shape) == 3:
                filter_size = text_feats.shape[0]
                if filter_size == 1:
                    ## single text - no support relations
                    similarity = torch.einsum('bcx, bc -> bx', image_feats, text_feats)
                    similarity = (torch.max(similarity, axis=0)[0]).unsqueeze(0)
                    return similarity

                image_feats = image_feats.reshape(image_feats.shape[0], image_feats.shape[1], -1, num_layers)
                B, C, HW, L = image_feats.shape
                K = L - filter_size + 1
                text_feats = text_feats.permute(1, 0).unsqueeze(0)
                sim = torch.zeros((image_feats.shape[2:])).unsqueeze(0)
                for b in range(B):
                    x = image_feats[b].permute(1, 0, 2)   # (HW, C, L)
                    w = text_feats[b].unsqueeze(1)
                    y = F.conv1d(x, w, bias=None, groups=C)
                    sim[b, :, :K] = y.sum(dim=1)

                sim = sim.reshape(B, -1)
            else:
                text_feats = text_feats.permute(1, 0).unsqueeze(0)
                sim = torch.zeros((image_feats.shape[2:])).unsqueeze(0)
                for k in range(image_feats.shape[-1]-1):
                    sim[:, :, :, k] = torch.einsum('bchwl, bcl -> bhw', image_feats[:, :, :, :, k:k+filter_height], text_feats)
            return sim

        if len(image_feats.shape) == 3:
            similarity = torch.einsum('bcx, bc -> bx', image_feats, text_feats)
        elif len(image_feats.shape) == 4:
            similarity = torch.einsum('bchw, bc -> bhw', image_feats, text_feats)
        else:
            similarity = torch.einsum('bchwl, bc -> bhwl', image_feats, text_feats)
        
        if self.fuse_similarity == "mean":
            similarity = torch.mean(similarity, axis=0).unsqueeze(0)
        elif self.fuse_similarity == "max":
            similarity = (torch.max(similarity, axis=0)[0]).unsqueeze(0)
        else:
            # "sum"
            similarity = torch.sum(similarity, axis=0).unsqueeze(0)
        return similarity

    # def forward(self, images: np.ndarray):
    #    return self.image_forward_torch(images)

    # def forward(self, text_tokenized: torch.Tensor):
    # print(text_tokenized.shape)
    # with torch.no_grad():
    # class_embeddings = self.clip_model.encode_text(text_tokenized)
    # return F.normalize(class_embeddings, dim=1)

    def forward_im(self, images: torch.Tensor):
        return self.image_forward_torch(images)

    def forward_text(self, text_tokenized):
        with torch.no_grad():
            class_embeddings = self.clip_model.encode_text(text_tokenized)
            return F.normalize(class_embeddings, dim=1)

    # def forward_text_trt(self, text_tokenized):
    #
    #
    #
    # #class_embeddings = self.clip_model.encode_text(text_tokenized)
    #
    #
    #
    #
    # return F.normalize(torch.tensor(output), dim=1)

    def image_forward_torch(self, clip_images: torch.Tensor):
        with torch.no_grad():
            clip_features = self.clip_model.encode_image(clip_images, dense=False)
            return F.normalize(clip_features, dim=1)

    def text_forward_trt(self, texts: torch.Tensor):
        # print(texts)
        output_shape = self.engine_txt.get_binding_shape(1)
        # output = np.empty(self.engine_txt.get_binding_shape(1), dtype=np.float32)
        input_tensor = texts.cuda()
        output_tensor = torch.empty(*output_shape, dtype=torch.float32, device="cuda")
        d_input = input_tensor.data_ptr()
        d_output = output_tensor.data_ptr()

        self.context_txt.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=self.stream.cuda_stream)
        # cuda.memcpy_dtoh_async(output, d_output, self.stream_txt)
        # self.stream_txt.synchronize()
        # d_output.free()
        # d_input.free()
        # print(output[:, 0])
        # print(output.sum())
        return F.normalize(output_tensor, dim=1).to(torch.float32)

    def image_forward_trt(self, input_tensor: torch.Tensor):
        # print(f"Input shape: {images.shape}, dtype: {images.dtype}")
        # print(f"Input binding shape: {self.engine.get_binding_shape(0)}")
        # print(f"Output binding shape: {self.engine.get_binding_shape(1)}")
        # TODO: likely the amount of needed copies can be reduced here
        # input_tensor = torch.tensor(images, dtype=torch.float32, device="cuda")

        output_shape = self.engine.get_binding_shape(1)
        output_tensor = torch.empty(*output_shape, dtype=torch.float32, device="cuda")
        d_output = output_tensor.data_ptr()
        d_input = input_tensor.data_ptr()
        # cuda.memcpy_htod_async(d_input, images , torch.cuda.current_stream())
        bindings = [int(d_input)] + [int(d_output)]
        # self.cfx.push()
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.cuda_stream)
        # self.cfx.pop()
        # cuda.memcpy_dtoh_async(output, d_output, torch.cuda.current_stream().current_stream)
        # self.stream.synchronize()
        # d_output.free()
        # d_input.free()
        return F.normalize(output_tensor, dim=1).to(torch.float32)

    def get_image_features(self,
                           images: np.ndarray
                           ) -> torch.Tensor:
        images = images[0]
        images = np.transpose(images, (1,2,0))
        with torch.no_grad():
            # if input type is tensor, no need to move anything to cpu
            input_t = type(images)
            if isinstance(images, np.ndarray):
                images = self.lseg_transform(images).unsqueeze(0).to(self.device)
            else:
                images = TS.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])(
                    images.permute(2, 0, 1) / 255
                ).unsqueeze(0)

            batch,_, h, w = images.size()
            stride_rate = 2.0 / 3.0
            stride = int(self.crop_size * stride_rate)

            # long_size = int(math.ceil(base_size * scale))
            long_size = self.base_size
            if h > w:
                height = long_size
                width = int(1.0 * w * long_size / h + 0.5)
                short_size = width
            else:
                width = long_size
                height = int(1.0 * h * long_size / w + 0.5)
                short_size = height

            cur_img = resize_image(
                images, height, width, **{"mode": "bilinear", "align_corners": True}
            )

            if long_size <= self.crop_size:
                pad_img = pad_image(cur_img, self.norm_mean, self.norm_std, self.crop_size)
                # print(pad_img.shape)
                with torch.no_grad():
                    # outputs = model(pad_img)
                    outputs, logits = self.lseg_model(pad_img, self.labels)
                outputs = crop_image(outputs, 0, height, 0, width)
            else:
                if short_size < self.crop_size:
                    # pad if needed
                    pad_img = pad_image(cur_img, self.norm_mean, self.norm_std, self.crop_size)
                else:
                    pad_img = cur_img
                _, _, ph, pw = pad_img.shape  # .size()
                assert ph >= height and pw >= width
                h_grids = int(math.ceil(1.0 * (ph - self.crop_size) / stride)) + 1
                w_grids = int(math.ceil(1.0 * (pw - self.crop_size) / stride)) + 1
                with torch.cuda.device_of(images):
                    with torch.no_grad():
                        outputs = (
                            images.new()
                            .resize_(batch, self.lseg_model.out_c, ph, pw)
                            .zero_()
                            .to(self.device)
                        )
                        logits_outputs = (
                            images.new()
                            .resize_(
                                batch,
                                len(self.labels) if input_t != torch.Tensor else self.labels.shape[0],
                                ph,
                                pw,
                            )
                            .zero_()
                            .to(self.device)
                        )
                    count_norm = (
                        images.new().resize_(batch, 1, ph, pw).zero_().to(self.device)
                    )
                # grid evaluation
                for idh in range(h_grids):
                    for idw in range(w_grids):
                        h0 = idh * stride
                        w0 = idw * stride
                        h1 = min(h0 + self.crop_size, ph)
                        w1 = min(w0 + self.crop_size, pw)
                        crop_img = crop_image(pad_img, h0, h1, w0, w1)
                        # pad if needed
                        pad_crop_img = pad_image(crop_img, self.norm_mean, self.norm_std, self.crop_size)
                        with torch.no_grad():
                            # output = model(pad_crop_img)
                            output, logits = self.lseg_model(pad_crop_img, self.labels)
                        cropped = crop_image(output, 0, h1 - h0, 0, w1 - w0)
                        cropped_logits = crop_image(logits, 0, h1 - h0, 0, w1 - w0)
                        outputs[:, :, h0:h1, w0:w1] += cropped
                        logits_outputs[:, :, h0:h1, w0:w1] += cropped_logits
                        count_norm[:, :, h0:h1, w0:w1] += 1
                assert (count_norm == 0).sum() == 0
                outputs = outputs / count_norm
                outputs = outputs[:, :, :height, :width]
            # outputs = resize_image(outputs, h, w, **{'mode': 'bilinear', 'align_corners': True})
            # outputs = resize_image(outputs, image.shape[0], image.shape[1], **{'mode': 'bilinear', 'align_corners': True})
            if input_t == np.ndarray:
                outputs = outputs.cpu()
                outputs = outputs.numpy()  # B, D, H, W
            
        outputs = torch.tensor(outputs, device=self.device).squeeze(0)
        return outputs

    def get_text_features(self,
                          texts: List[str]
                          ) -> torch.Tensor:
        with torch.no_grad():
            texts = self.tokenizer(texts)

            # print(texts.shape)
            # print(texts.dtype)
            # After, differentiate between jetson and normal

            if self.jetson:
                return self.text_forward_trt(texts)
            else:
                class_embeddings = self.clip_model.encode_text(texts.to("cuda"))
                return F.normalize(class_embeddings, dim=1)


if __name__ == "__main__":
    import time
    use_jetson = False
    N = 1
    import cv2
    
