# numpy
import numpy as np

# modeling
from vision_models.base_model import BaseModel
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T
from torch.nn import functional as F
from fvcore.common.checkpoint import Checkpointer
import open_clip
import torch

# Visualization
import matplotlib.pyplot as plt

# typing
from typing import List, Optional

from PIL import Image

try:
    import tensorrt as trt
except:
    print("TensorRT not available, cannot use Jetson")

class ClipImageModel(torch.nn.Module, BaseModel):
    def __init__(self,
                 jetson: bool = False,
                 fuse_similarity: str = "sum",
                 ):
        super(ClipImageModel, self).__init__()
        self.jetson = jetson
        self.fuse_similarity = fuse_similarity

        self.input_format = "RGB"

        self.aug = T.ResizeShortestEdge(
            [640, 640], 2560
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.feature_dim = 512
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
        # expects images in shape B C H W in BGR, expected to be a numpy array
        with torch.no_grad():
            # Apply pre-processing to image.
            if self.input_format == "BGR":
                # whether the model expects BGR inputs or RGB
                original_image = images[:, ::-1, :, :]
            else:
                original_image = images
            
            transformed = []
            for b in range(original_image.shape[0]):
                transformed.append(self.clip_preprocess(Image.fromarray(np.transpose(original_image[b], (1,2,0)))))

            transformed = torch.tensor(np.stack(transformed), device="cuda")
            image_feats = self.image_forward_torch(transformed)
            image_feats = image_feats.unsqueeze(-1).unsqueeze(-1).squeeze(0)
            return image_feats

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
    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    clip = ClipModel('../weights/clip.pth', use_jetson) # Jetson
    img = read_image('rgb.jpg', format="RGB")
    # img = read_image('/home/finn/drafting/CLIPTest/sim2.png', format="RGB")
    # img = read_image('/home/Pictures/chair.png', format="RGB")
    # img = read_image('/home/spot/chair.png', format="RGB")
    # img = cv2.resize(img, (640, 640))
    img = img.transpose(2, 0, 1)
    img_feats_ = clip.get_image_features(img)
    # print(img_feats_)
    # text_feats = clip.encode_text("A photo of a robot endeffector")

    # start.record()
    # Perform N iterations and measure overall time
    print("a")
    start_time = time.time()
    for i in range(N):
        # img[:, i*5:(i+1*5)] -= i
        img_feats = clip.get_image_features(img)
        # print(img_feats.sum())
        # print(img_feats.sum())
        torch.cuda.synchronize()  # Synchronize after each forward pass

    end_time = time.time()
    # Compute overall time and average time per iteration
    total_time = end_time - start_time
    avg_time_per_iteration = total_time / N
    # end.record()

    print(f"Total time for {N} iterations: {total_time:.4f} seconds")
    print(f"Average time per iteration: {avg_time_per_iteration:.4f} seconds")    # clip = ClipModel('weights/clip.pth', False) # Jetson
    txt_feats = clip.get_text_features(["a chair"])
    sim = clip.compute_similarity(img_feats, txt_feats)
    print(sim.max(), sim.min(), sim.mean())
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(sim[0].detach().cpu())
    axs[1].imshow(img.transpose(1, 2, 0))
    plt.savefig("plant.png")
    plt.show()
    # print(img_feats.shape, text_feats.shape)
