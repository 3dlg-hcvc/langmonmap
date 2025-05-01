# numpy
import os
import imageio
import numpy as np
import torch
import textwrap

# cv2
import cv2

# rerun
import rerun as rr

from typing import (
    Dict,
    List,
    Optional,
    Union,
)
from habitat.utils.visualizations.utils import tile_images
import supervision as sv
from PIL import Image

SENSORS_TO_INCLUDE = ["rgb"]

def log_map_rerun(map_, path, needs_orientation=False):
    """
    Applies the inferno colormap to the map and logs it to rerun at the given path
    :param map_: 2D array
    :param path: logging path
    :param needs_orientation:
    :return:
    """
    if needs_orientation:
        map_ = map_.transpose((1, 0))
        map_ = np.flip(map_, axis=0)
    map_ = monochannel_to_inferno_rgb(map_)
    rr.log(path, rr.Image(np.flip(map_, axis=-1)).compress(jpeg_quality=50))


def publish_sim_map(sim_map, br, publisher):
    sim_map = sim_map.transpose((1, 0))
    sim_map = np.flip(sim_map, axis=0)
    sim_map = monochannel_to_inferno_rgb(sim_map)
    # upscale to 1000x1000
    sim_map = cv2.resize(sim_map, (1000, 1000))
    img_msg = br.cv2_to_imgmsg(sim_map, encoding="bgr8")
    publisher.publish(img_msg)

def monochannel_to_inferno_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a monochannel float32 image to an RGB representation using the Inferno
    colormap.

    Args:
        image (numpy.ndarray): The input monochannel float32 image.

    Returns:
        numpy.ndarray: The RGB image with Inferno colormap.
    """
    # Normalize the input image to the range [0, 1]
    min_val, max_val = np.min(image), np.max(image)
    peak_to_peak = max_val - min_val
    if peak_to_peak == 0:
        normalized_image = np.zeros_like(image)
    else:
        normalized_image = (image - min_val) / peak_to_peak

    # Apply the Inferno colormap
    inferno_colormap = cv2.applyColorMap((normalized_image * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

    return inferno_colormap

def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_")

    # File names are not allowed to be over 255 characters
    video_name_split = video_name.split("/")
    video_name = "/".join(
        video_name_split[:-1] + [video_name_split[-1][:251] + ".mp4"]
    )

    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
    )
    images_iter: List[np.ndarray] = images
    for im in images_iter:
        try:
            writer.append_data(im)
        except:
            im1=cv2.resize(im, (images[0].shape[1], images[0].shape[0]))
            writer.append_data(im1)
    writer.close()

def generate_video(
    video_dir: Optional[str],
    video_name: str,
    images: List[np.ndarray],
    fps: int = 10,
    verbose: bool = True,
) -> None:
    r""" similar to habitat_baselines.utils.common.generate_video

    Args:
        video_dir: path to target video directory.
        images: list of images to be converted to video.
        fps: fps for generated video.
    Returns:
        None
    """
    if len(images) < 1:
        return

    images_to_video(
        images, video_dir, video_name, fps=fps, verbose=verbose
    )

def append_text_underneath_image(image: np.ndarray, text: str):
    """Appends text underneath an image of size (height, width, channels).

    The returned image has white text on a black background. Uses textwrap to
    split long text into multiple lines.

    :param image: The image to appends text underneath.
    :param text: The string to display.
    :return: A new image with text appended underneath.
    """
    h, w, c = image.shape
    font_size = 1.5
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    char_size = cv2.getTextSize(" ", font, font_size, font_thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            (255, 255, 255),
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    text_image = blank_image[0 : y + 10, 0:w]
    final = np.concatenate((image, text_image), axis=0)
    return final

def add_sim_maps_to_image(observation: Dict, maps: Dict=None, info: Dict=None, text_to_append: str = "") -> np.ndarray:

    render_obs_images: List[np.ndarray] = []
    if (
        len(maps) > 0
        and "object_detected" in maps
        and maps["object_detected"]
        and "predictions" in maps
        and len(maps["predictions"]) > 0
        and len(maps["predictions"]["boxes"]) > 0
    ):
        predictions = maps["predictions"]
        detections = sv.Detections(
            xyxy=np.array(predictions['boxes']),
            class_id=np.array([0 for _ in range(len(predictions['boxes']))]),
            confidence=np.array(predictions['scores'])
        )

        obs_k = observation["rgb"][:,:,:3]
        if not isinstance(obs_k, np.ndarray):
            obs_k = obs_k.cpu().numpy()
        svimage = Image.fromarray(obs_k)

        bounding_box_annotator = sv.BoxAnnotator()
        svimage = bounding_box_annotator.annotate(svimage, detections)
        render_obs_images.append(np.array(svimage))
    else:
        for sensor_name in observation:
            if sensor_name not in SENSORS_TO_INCLUDE:
                continue

            if isinstance(observation[sensor_name], np.ndarray) and len(observation[sensor_name].shape) > 1:
                obs_k = observation[sensor_name]
                if not isinstance(obs_k, np.ndarray):
                    obs_k = obs_k.cpu().numpy()
                if obs_k.dtype != np.uint8:
                    obs_k = obs_k * 255.0
                    obs_k = obs_k.astype(np.uint8)
                if len(obs_k.shape) == 3 and obs_k.shape[2] == 4:
                    obs_k = obs_k[:, :, :3]
                if len(maps) > 0 and "draw_found_on_map" in maps:
                    goal_center = maps["draw_found_on_map"]
                    start_y, end_y, = goal_center[0] - 10, goal_center[0] + 10
                    start_x, end_x, = goal_center[1] - 10, goal_center[1] + 10
                    obs_k = cv2.rectangle(obs_k.astype(np.uint8), (start_x, start_y), (end_x, end_y), (128, 128, 0), thickness=2)

                if sensor_name == "depth":
                    mask = obs_k == float('inf')
                    obs_k[mask] = obs_k[~mask].max()
                    kernel_size = 11
                    pad = kernel_size // 2

                    depth_image_smoothed = -torch.nn.functional.max_pool2d(torch.from_numpy(-obs_k).unsqueeze(0), kernel_size,
                                                                        padding=pad,
                                                                        stride=1).squeeze(0)
                    obs_k = depth_image_smoothed.cpu().numpy()
                    if len(obs_k.shape) == 2:
                        obs_k = obs_k[..., np.newaxis]
                    if obs_k.shape[2] == 1:
                        obs_k = np.concatenate([obs_k for _ in range(3)], axis=2)
                render_obs_images.append(obs_k)

    shapes_are_equal = len(set(x.shape for x in render_obs_images)) == 1
    if not shapes_are_equal:
        render_frame = tile_images(render_obs_images)
    else:
        render_frame = np.concatenate(render_obs_images, axis=1)

    ## add sim maps to the frame
    if len(maps) > 0:
        if "sim_map" in maps and maps["sim_map"] is not None:
            _image = maps["sim_map"]
            old_h, old_w, _ = _image.shape
            img_height = render_frame.shape[0]
            img_width = int(float(img_height) / old_h * old_w)
            # cv2 resize (dsize is width first)
            _image = cv2.resize(
                _image,
                (img_width, img_height),
                interpolation=cv2.INTER_CUBIC,
            )
            render_frame = np.concatenate((render_frame, _image), axis=1)
        if "sim_map_layers" in maps:
            _images = maps["sim_map_layers"]
            font_size = 1.5
            font_thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, _image in enumerate(_images):
                old_h, old_w, _ = _image.shape
                img_height = render_frame.shape[0]
                img_width = int(float(img_height) / old_h * old_w)
                # cv2 resize (dsize is width first)
                _image = cv2.resize(
                    _image,
                    (img_width, img_height),
                    interpolation=cv2.INTER_CUBIC,
                )
                cv2.putText(
                    _image,
                    f"layer {i}",
                    (img_width-200, img_height-50),
                    font,
                    font_size,
                    (0, 0, 0),
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )
                render_frame = np.concatenate((render_frame, _image), axis=1)
        
        next_layer_frames = []
        if "obstcl_map_layers" in maps and maps["obstcl_map_layers"] is not None:
            if "traversable_map" in maps:
                _image = maps["traversable_map"]
                old_h, old_w, _ = _image.shape
                img_height = render_frame.shape[0]
                img_width = int(float(img_height) / old_h * old_w)
                # cv2 resize (dsize is width first)
                _image = cv2.resize(
                    _image,
                    (img_width, img_height),
                    interpolation=cv2.INTER_CUBIC,
                )
                cv2.putText(
                    _image,
                    f"navigable map",
                    (img_width-200, img_height-50),
                    font,
                    font_size,
                    (0, 0, 0),
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )
                next_layer_frames.append(_image)

            _images = maps["obstcl_map_layers"]
            font_size = 1.5
            font_thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i, _image in enumerate(_images):
                old_h, old_w, _ = _image.shape
                img_height = render_frame.shape[0]
                img_width = int(float(img_height) / old_h * old_w)
                # cv2 resize (dsize is width first)
                _image = cv2.resize(
                    _image,
                    (img_width, img_height),
                    interpolation=cv2.INTER_CUBIC,
                )
                cv2.putText(
                    _image,
                    f"layer {i}",
                    (img_width-200, img_height-50),
                    font,
                    font_size,
                    (250, 250, 250),
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )
                next_layer_frames.append(_image)

            shapes_are_equal = len(set(x.shape for x in next_layer_frames)) == 1
            if not shapes_are_equal:
                render_frame_next = tile_images(next_layer_frames)
            else:
                render_frame_next = np.concatenate(next_layer_frames, axis=1)

            render_frame = np.concatenate((render_frame, render_frame_next), axis=0)
        else:
            if "traversable_map" in maps:
                _image = maps["traversable_map"]
                old_h, old_w, _ = _image.shape
                img_height = render_frame.shape[0]
                img_width = int(float(img_height) / old_h * old_w)
                # cv2 resize (dsize is width first)
                _image = cv2.resize(
                    _image,
                    (img_width, img_height),
                    interpolation=cv2.INTER_CUBIC,
                )
                render_frame = np.concatenate((render_frame, _image), axis=1)

    ## append text
    if len(text_to_append) > 0:
        render_frame = append_text_underneath_image(render_frame, text_to_append)

    return render_frame
