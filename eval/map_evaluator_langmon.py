# eval utils
import argparse
from eval import get_closest_dist, FMMPlanner
from eval.actor import Actor
from onemap_utils import monochannel_to_inferno_rgb
from eval.dataset_utils import *
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from textwrap import wrap
from sklearn.cluster import DBSCAN

# os / filsystem
import os
from os import listdir
import tqdm

import cv2

# numpy
import numpy as np
import torch

from vision_models.clip_dense import ClipModel
from vision_models.yolo_world_detector import YOLOWorldDetector
import supervision as sv
from PIL import Image

from vision_models.yolov7_model import YOLOv7Detector

SEQ_LEN = 1
results_path = "results_langmon_detailed_expnexp"
object_nav_path = "datasets/langmon/langmon/fphab/v0.2/detailed" #"datasets/langmon/minival_full/minival/"
path_evaluation = "map_evaluation"
device = "cuda"
include_episodes = [
    # "106879005_174887124__1",
]
state_path = "state"


def evaluate_maps_on_custom_prompt(args):

    query_argument = args.query
    sampling_method = args.sampling_method

    generated_map_dir = os.path.join(results_path, "saved_maps")
    # gt_map_dir = os.path.join(results_path, "saved_maps_gt")

    # Check if the state directory exists
    if not os.path.isdir(generated_map_dir):
        print(f"Error: {generated_map_dir} is not a valid directory")

    # load episode details
    scene_data = {}
    episodes = []
    episodes, scene_data = LangMonDataset.load_hm3d_multi_episodes(
        episodes, scene_data, object_nav_path
    )

    # Iterate through all files in the directory
    for filename in sorted(listdir(generated_map_dir)):

        # load generated maps
        map_objects = np.load(
            os.path.join(generated_map_dir, filename), allow_pickle=True
        )
        navigable_map = map_objects["nav_map"]
        feature_map = map_objects["feature_map"]
        confidence_map = map_objects["confidence_map"]
        query = (
            str(map_objects["query"]) if len(query_argument) == 0 else query_argument
        )
        pose_observations = map_objects["pose_observations"]
        gt_goal_objects = map_objects["gt_goal_objects"]

        evaluation_dir = os.path.join(results_path, path_evaluation, query)
        os.makedirs(evaluation_dir, exist_ok=True)

        # load ground-truth maps
        # gt_map_objects = np.load(os.path.join(gt_map_dir, filename), allow_pickle=True)
        # gt_topdown_map = gt_map_objects["gt_topdown_map"]
        # gt_object_goals = gt_map_objects["gt_object_goals"]
        # experiment_result = gt_map_objects["experiment_result"][0]
        # is_success = "Y" if int(experiment_result["state"]) == 1 else "N"
        # goal_sequence = experiment_result["sequence"]
        # experiment_num = experiment_result["experiment"]

        # find similarity map
        feature_map_tensor = (
            torch.from_numpy(feature_map)
            .type(torch.float)
            .to(device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        clip = ClipModel("weights/clip.pth")
        query_text_features = clip.get_text_features([query]).to(device)
        similarity = clip.compute_similarity(feature_map_tensor, query_text_features)
        # similarity[similarity<0] = 0
        final_sim = ((similarity + 1.0) / 2.0).cpu().numpy()
        final_sim = final_sim[0]

        # save as image
        final_sim_img = monochannel_to_inferno_rgb(final_sim)
        final_sim_img = final_sim_img.transpose((1, 0, 2))
        final_sim_img = np.flip(final_sim_img, axis=0)

        poses_ = np.array([po["pose_map"] for po in pose_observations[-3:]])

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(
            final_sim_img[:, :, ::-1],
            interpolation="nearest",
            aspect="equal",
            extent=(0, final_sim_img.shape[1], 0, final_sim_img.shape[0]),
        )

        # Plot end of trajectory
        px, py = pose_observations[-1]["pose_map"]
        ax1.plot(px, py, "gs")
        u = np.diff(poses_[:, 0])
        v = np.diff(poses_[:, 1])
        pos_x = poses_[:, 0][:-1] + u / 2
        pos_y = poses_[:, 1][:-1] + v / 2
        norm = np.sqrt(u**2 + v**2)
        ax1.quiver(
            pos_x, pos_y, u / norm, v / norm, angles="xy", pivot="mid", color="g"
        )
        # ax1.plot(np.array([px,facing_px]), np.array([py,facing_py]), "g-")

        # Plot sampled goal locations
        if sampling_method == "percentile":
            ## top k percentile
            percentile_ = args.percentile_q_value
            top_ = np.percentile(final_sim, percentile_)
            top_goals = np.transpose(np.where(final_sim > top_))
            for g in top_goals:
                ax1.plot(g[0], g[1], "r+")
            ax2.set_title(
                "\n".join(
                    wrap(
                        f"Prompt:{query}; red+: top-{100-percentile_} percentile={top_}"
                    )
                )
            )
        elif sampling_method == "highest_score":
            ## highest score
            highest_score = np.max(final_sim)
            goal_w_highest_score = np.unravel_index(
                np.argmax(final_sim), final_sim.shape
            )
            ax1.plot(goal_w_highest_score[0], goal_w_highest_score[1], "r*")
            ax2.set_title(
                "\n".join(
                    wrap(f"Prompt:{query}; red*: goal w/ highest score={highest_score}")
                )
            )
        elif sampling_method == "score_threshold":
            ## score above a threshold
            score_threshold_value = args.score_threshold_value
            top_goals = np.transpose(np.where(final_sim > score_threshold_value))
            for g in top_goals:
                ax1.plot(g[0], g[1], "r+")
            ax2.set_title(
                "\n".join(
                    wrap(
                        f"Prompt:{query}; red+: thesholded at score {score_threshold_value}"
                    )
                )
            )
        elif sampling_method == "cluster":
            ## cluster scores
            score_threshold_value = args.score_threshold_value
            important_pixels = np.transpose(np.where(final_sim > score_threshold_value))
            try:
                db = DBSCAN(eps=2, min_samples=5).fit(important_pixels)
                ### Extract cluster labels and find centers
                labels = db.labels_
                unique_labels = set(labels)
                centers = []

                for label in unique_labels:
                    if label == -1:  # -1 means noise in DBSCAN, ignore it
                        continue
                    cluster_points = important_pixels[labels == label]
                    center = cluster_points.mean(axis=0)
                    centers.append(np.array(center, dtype=np.uint8))
                    ax1.plot(int(center[0]), int(center[1]), "b+")
                    t = ax1.text(
                        int(center[0]) - 10,
                        int(center[1]) - 10,
                        "%s" % (label),
                        color="white",
                    )
                    t.set_ha("center")

                ax2.set_title(
                    "\n".join(
                        wrap(
                            f"Prompt:{query}; blue+: {len(centers)} cluster centers for scores > {score_threshold_value}"
                        )
                    )
                )
            except:
                pass

        # Set equal aspect ratio to ensure accurate positions
        ax1.axis("equal")

        # Add labels and title
        ax1.set_xlabel("X position")
        ax1.set_ylabel("Y position")
        ax1.set_title("Topdown 2D Feature Map")

        # Add grid for better readability
        ax1.grid(True)

        ax2.imshow(pose_observations[-1]["obs_from_pose"]["rgb"])
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Save the plot as SVG
        plt.savefig(
            f"{evaluation_dir}/{filename[:-4]}_{sampling_method}.svg",
            format="svg",
            dpi=300,
            bbox_inches="tight",
        )


def evaluate_maps_on_custom_prompt_by_nouns(args):

    query_argument = args.query
    sampling_method = args.sampling_method

    generated_map_dir = os.path.join(results_path, "saved_maps")
    gt_map_dir = os.path.join(results_path, "saved_maps_gt")

    # Check if the state directory exists
    if not os.path.isdir(generated_map_dir):
        print(f"Error: {generated_map_dir} is not a valid directory")

    # load episode details
    scene_data = {}
    episodes = []
    episodes, scene_data = LangMonDataset.load_hm3d_multi_episodes(
        episodes, scene_data, object_nav_path
    )

    # Iterate through all files in the directory
    for filename in sorted(listdir(generated_map_dir)):

        # load generated maps
        map_objects = np.load(
            os.path.join(generated_map_dir, filename), allow_pickle=True
        )
        navigable_map = map_objects["nav_map"]
        feature_map = map_objects["feature_map"]
        confidence_map = map_objects["confidence_map"]
        query = (
            str(map_objects["query"]) if len(query_argument) == 0 else query_argument
        )
        pose_observations = map_objects["pose_observations"]
        gt_goal_objects = map_objects["gt_goal_objects"]

        evaluation_dir = os.path.join(
            results_path, path_evaluation, query, "nouns_parsed"
        )
        os.makedirs(evaluation_dir, exist_ok=True)

        # load ground-truth maps
        gt_map_objects = np.load(os.path.join(gt_map_dir, filename), allow_pickle=True)
        gt_topdown_map = gt_map_objects["gt_topdown_map"]
        gt_object_goals = gt_map_objects["gt_object_goals"]
        experiment_result = gt_map_objects["experiment_result"][0]
        is_success = "Y" if int(experiment_result["state"]) == 1 else "N"
        goal_sequence = experiment_result["sequence"]
        experiment_num = experiment_result["experiment"]

        # find similarity map
        feature_map_tensor = (
            torch.from_numpy(feature_map)
            .type(torch.float)
            .to(device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        clip = ClipModel("weights/clip.pth")
        query_text_features = clip.get_text_features(["a couch"]).to(device)
        similarity1 = clip.compute_similarity(feature_map_tensor, query_text_features)
        query_text_features = clip.get_text_features(["a fireplace"]).to(device)
        similarity2 = clip.compute_similarity(feature_map_tensor, query_text_features)
        similarity = (similarity1 + similarity2) / 2.0
        final_sim = ((similarity + 1.0) / 2.0).cpu().numpy()
        final_sim = final_sim[0]

        # save as image
        final_sim_img = monochannel_to_inferno_rgb(final_sim)
        final_sim_img = final_sim_img.transpose((1, 0, 2))
        final_sim_img = np.flip(final_sim_img, axis=0)

        poses_ = np.array([po["pose_map"] for po in pose_observations[-3:]])

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.imshow(
            final_sim_img[:, :, ::-1],
            interpolation="nearest",
            aspect="equal",
            extent=(0, final_sim_img.shape[1], 0, final_sim_img.shape[0]),
        )

        # Plot end of trajectory
        px, py = pose_observations[-1]["pose_map"]
        ax1.plot(px, py, "gs")
        u = np.diff(poses_[:, 0])
        v = np.diff(poses_[:, 1])
        pos_x = poses_[:, 0][:-1] + u / 2
        pos_y = poses_[:, 1][:-1] + v / 2
        norm = np.sqrt(u**2 + v**2)
        ax1.quiver(
            pos_x, pos_y, u / norm, v / norm, angles="xy", pivot="mid", color="g"
        )
        # ax1.plot(np.array([px,facing_px]), np.array([py,facing_py]), "g-")

        # Plot sampled goal locations
        if sampling_method == "percentile":
            ## top k percentile
            percentile_ = args.percentile_q_value
            top_ = np.percentile(final_sim, percentile_)
            top_goals = np.transpose(np.where(final_sim > top_))
            for g in top_goals:
                ax1.plot(g[0], g[1], "r+")
            ax2.set_title(
                "\n".join(
                    wrap(
                        f"Prompt:{query}; red+: top-{100-percentile_} percentile={top_}"
                    )
                )
            )
        elif sampling_method == "highest_score":
            ## highest score
            highest_score = np.max(final_sim)
            goal_w_highest_score = np.unravel_index(
                np.argmax(final_sim), final_sim.shape
            )
            ax1.plot(goal_w_highest_score[0], goal_w_highest_score[1], "r*")
            ax2.set_title(
                "\n".join(
                    wrap(f"Prompt:{query}; red*: goal w/ highest score={highest_score}")
                )
            )
        elif sampling_method == "score_threshold":
            ## score above a threshold
            score_threshold_value = args.score_threshold_value
            top_goals = np.transpose(np.where(final_sim > score_threshold_value))
            for g in top_goals:
                ax1.plot(g[0], g[1], "r+")
            ax2.set_title(
                "\n".join(
                    wrap(
                        f"Prompt:{query}; red+: thesholded at score {score_threshold_value}"
                    )
                )
            )
        elif sampling_method == "cluster":
            ## cluster scores
            score_threshold_value = args.score_threshold_value
            important_pixels = np.transpose(np.where(final_sim > score_threshold_value))
            try:
                db = DBSCAN(eps=2, min_samples=5).fit(important_pixels)
                ### Extract cluster labels and find centers
                labels = db.labels_
                unique_labels = set(labels)
                centers = []

                for label in unique_labels:
                    if label == -1:  # -1 means noise in DBSCAN, ignore it
                        continue
                    cluster_points = important_pixels[labels == label]
                    center = cluster_points.mean(axis=0)
                    centers.append(np.array(center, dtype=np.uint8))
                    ax1.plot(int(center[0]), int(center[1]), "b+")
                    t = ax1.text(
                        int(center[0]) - 10,
                        int(center[1]) - 10,
                        "%s" % (label),
                        color="white",
                    )
                    t.set_ha("center")

                ax2.set_title(
                    "\n".join(
                        wrap(
                            f"Prompt:{query}; blue+: {len(centers)} cluster centers for scores > {score_threshold_value}"
                        )
                    )
                )
            except:
                pass

        # Set equal aspect ratio to ensure accurate positions
        ax1.axis("equal")

        # Add labels and title
        ax1.set_xlabel("X position")
        ax1.set_ylabel("Y position")
        ax1.set_title("Topdown 2D Feature Map")

        # Add grid for better readability
        ax1.grid(True)

        ax2.imshow(pose_observations[-1]["obs_from_pose"]["rgb"])
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Save the plot as SVG
        plt.savefig(
            f"{evaluation_dir}/{filename[:-4]}_{sampling_method}.svg",
            format="svg",
            dpi=300,
            bbox_inches="tight",
        )


def evaluate_maps():

    generated_map_dir = os.path.join(results_path, "saved_maps")
    # gt_map_dir = os.path.join(results_path, "saved_maps_gt")
    state_dir = os.path.join(results_path, state_path)

    # Check if the state directory exists
    if not os.path.isdir(generated_map_dir):
        print(f"Error: {generated_map_dir} is not a valid directory")

    # load episode details
    scene_data = {}
    episodes = []
    episodes, scene_data = LangMonDataset.load_hm3d_multi_episodes(
        episodes, scene_data, object_nav_path
    )

    clip = ClipModel("weights/clip.pth")

    # Iterate through all files in the directory
    for filename in sorted(listdir(generated_map_dir)):

        experiment_num = filename[:-6]
        if len(include_episodes) > 0 and experiment_num not in include_episodes:
            continue

        # load generated maps
        map_objects = np.load(
            os.path.join(generated_map_dir, filename), allow_pickle=True
        )
        feature_map = map_objects["feature_map"]
        pose_observations = map_objects["pose_observations"]
        gt_goal_objects = map_objects["gt_goal_objects"]
        query = str(map_objects["query"]) #" ".join(gt_goal_objects[0]['object_extras']['object_category'].split("_"))

        # load states
        with open(os.path.join(state_dir, f"state_{experiment_num}.txt"), "r") as file:
            content = file.read().strip()

        state_values = content.split(",")
        state_values = [int(val) for val in state_values]
        is_success = "Y" if int(state_values[0]) == 1 else "N"
        goal_sequence = 0

        # load ground-truth maps
        # gt_map_objects = np.load(os.path.join(gt_map_dir, filename), allow_pickle=True)
        # gt_topdown_map = gt_map_objects["gt_topdown_map"]
        # gt_object_goals = gt_map_objects["gt_object_goals"]
        # experiment_result = gt_map_objects["experiment_result"][0]
        # goal_sequence = experiment_result["sequence"]

        # find similarity map
        feature_map_tensor = (
            torch.from_numpy(feature_map)
            .type(torch.float)
            .to(device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        query_text_features = clip.get_text_features(["a " + query]).to(device)
        similarity = clip.compute_similarity(feature_map_tensor, query_text_features)
        # similarity[similarity<0] = 0
        final_sim = ((similarity + 1.0) / 2.0).cpu().numpy()
        final_sim = final_sim[0]

        # save as image
        final_sim_img = monochannel_to_inferno_rgb(final_sim)
        final_sim_img = final_sim_img.transpose((1, 0, 2))
        final_sim_img = np.flip(final_sim_img, axis=0)

        # Create the plot
        fig = plt.figure(figsize=(40, 10))
        outer_grid = fig.add_gridspec(1, 2, width_ratios=[1, 4])

        # Add the three "normal" subplots
        ax1 = fig.add_subplot(outer_grid[0, 0])

        poses_ = np.array([po["pose_map"] for po in pose_observations])
        ax1.imshow(
            final_sim_img[:, :, ::-1],
            interpolation="nearest",
            aspect="equal",
            extent=(0, final_sim_img.shape[1], 0, final_sim_img.shape[0]),
        )

        ax1.plot(poses_[:, 0], poses_[:, 1], "b-")
        ax1.plot(poses_[0, 0], poses_[0, 1], "ro")
        ax1.plot(poses_[-1, 0], poses_[-1, 1], "go")

        # u = np.diff(poses_[:, 0])
        # v = np.diff(poses_[:, 1])
        # pos_x = poses_[:, 0][:-1] + u/2
        # pos_y = poses_[:, 1][:-1] + v/2
        # norm = np.sqrt(u**2+v**2)
        # ax1.quiver(pos_x, pos_y, u/norm, v/norm, angles="xy", zorder=5, pivot="mid")

        obj_locs = np.array([go["center_map"] for go in gt_goal_objects])
        obj_viewpts = np.array([go["nav_points_map"] for go in gt_goal_objects])
        for i, obj in enumerate(obj_locs):
            ax1.plot(obj[0], obj[1], "m.")
            for obj in obj_viewpts[i]:
                ax1.plot(obj[0], obj[1], "m.")

        # Set equal aspect ratio to ensure accurate positions
        ax1.axis("equal")

        # Add labels and title
        ax1.set_xlabel("X position")
        ax1.set_ylabel("Y position")
        ax1.set_title("Path of Poses")

        # Add grid for better readability
        ax1.grid(True)

        inner = gridspec.GridSpecFromSubplotSpec(3, 5, subplot_spec=outer_grid[1])
        for j in range(15):
            ax = plt.Subplot(fig, inner[j])
            t = ax.text(0.0, 0.0, "(T-%d)" % (j))
            t.set_ha("center")
            try:
                ax.imshow(pose_observations[-1 - j]["obs_from_pose"]["rgb"])
            except:
                break
            ax.set_xticks([])
            ax.set_yticks([])
            fig.add_subplot(ax)

        fig.suptitle(
            f"[Episode:{experiment_num}; Goal:({goal_sequence}){query}; Success:{is_success}]"
        )

        eval_dir = os.path.join(results_path, path_evaluation)
        os.makedirs(eval_dir, exist_ok=True)
        # Save the plot as SVG
        plt.savefig(
            f"{results_path}/{path_evaluation}/{is_success}_{experiment_num}_{query}_sim_map.svg",
            format="svg",
            dpi=300,
            bbox_inches="tight",
        )

        # cv2.imwrite(
        #     f"{results_path}/{path_evaluation}/{experiment_num}_sim_map.png",
        #     final_sim_img,
        # )

        # save topdown map
        # cv2.putText(
        #     gt_topdown_map,
        #     f'[Goals: {query}]',
        #     (10, 10),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.3,
        #     (0, 0, 0),
        #     1,
        #     lineType=cv2.LINE_AA,
        # )
        # cv2.imwrite(
        #     f"{results_path}/{path_evaluation}/{experiment_num}_gt_map.png",
        #     gt_topdown_map,
        # )

def test_similarity():
    testing_dir = 'testing'
    yolo_confidence = 0.6
    clip = ClipModel("weights/clip.pth")
    queries = [
        'the digital piano',
        # 'Find the chair in front of the table.',
        # 'Find the chair.',
        # 'Find the plant.',
        # 'Find the potted plant.',
        # 'Find the potted plant on the table.',
        # 'plant',
        # 'chair',
        # 'beanbag chair',
        # 'table'
    ]
    
    img_path = "/localhome/sraychau/Downloads/the digital piano.png"

    for query in queries:
    
        img = Image.open(img_path).convert('RGB')
        img_array = np.asarray(img).transpose(2,0,1)[np.newaxis, ...]
        query_text_features = clip.get_text_features([query]).to(device)
        image_features = clip.get_image_features(img_array).to(device)
        similarity = clip.compute_similarity(image_features, query_text_features).detach().cpu()

        print(similarity.max(), similarity.min(), similarity.mean())
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(similarity[0])
        axs[1].imshow(img_array.squeeze(0).transpose(1, 2, 0))
        axs[1].set_title(query)
        plt.savefig(os.path.join(testing_dir, f"{query}_similarity.jpg"))
        plt.show()

        ## Yolo v7 detection
        classes_ = [query]
        detector = YOLOv7Detector(yolo_confidence)
        detector.set_classes(classes_)

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

        # Load an image
        image = cv2.imread(img_path)

        # Detect objects in the image
        preds = detector.detect(image)

        if len(preds['boxes']) > 0:
            detections = sv.Detections(
                xyxy=np.array(preds['boxes']),
                class_id=np.array([0]),
                confidence=np.array(preds['scores'])
            )

            labels = [
                f"{classes_[class_id]} {confidence:0.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]

            image = Image.open(img_path)
            svimage = np.array(image)
            svimage = bounding_box_annotator.annotate(svimage, detections)
            svimage = label_annotator.annotate(svimage, detections, labels)
            
            sv.plot_image(svimage[:, :, ::-1])
            Image.fromarray(svimage[:, :, :-1]).save(os.path.join(testing_dir, f"{query}_yolov7.jpg"))
        else:
            image = np.array(Image.open(img_path))
            Image.fromarray(image[:,:,:-1]).save(os.path.join(testing_dir, f"{query}_yolov7.jpg"))

        ## Yolo world detection
        # Test the YOLO World Detector
        detector = YOLOWorldDetector(confidence_threshold=yolo_confidence)
        detector.set_classes(classes_)

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)

        # Load an image
        image = cv2.imread(img_path)

        # Detect objects in the image
        preds = detector.detect(image)

        if len(preds['boxes']) > 0:
            detections = sv.Detections(
                xyxy=np.array(preds['boxes']),
                class_id=np.array([0]),
                confidence=np.array(preds['scores'])
            )

            labels = [
                f"{classes_[class_id]} {confidence:0.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]

            image = Image.open(img_path)
            svimage = np.array(image)
            svimage = bounding_box_annotator.annotate(svimage, detections)
            svimage = label_annotator.annotate(svimage, detections, labels)
            
            sv.plot_image(svimage[:, :, ::-1])
            Image.fromarray(svimage[:, :, :-1]).save(os.path.join(testing_dir, f"{query}_yoloworld.jpg"))
        else:
            image = np.array(Image.open(img_path))
            Image.fromarray(image[:,:,:-1]).save(os.path.join(testing_dir, f"{query}_yoloworld.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="")
    parser.add_argument("--sampling_method", type=str, default="percentile")
    parser.add_argument("--percentile_q_value", type=float, default=95.0)
    parser.add_argument("--score_threshold_value", type=float, default=0.5)
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--process_all_sampling", type=str, default="Y")
    args = parser.parse_args()

    # test_similarity()

    evaluate_maps()

    # if args.process_all_sampling == "Y":
    #     all_sampling_methods = [
    #         "percentile",
    #         "cluster",
    #         "score_threshold",
    #         "highest_score",
    #     ]
    #     for m in tqdm.tqdm(all_sampling_methods):
    #         args.sampling_method = m
    #         evaluate_maps_on_custom_prompt(args)
    #         # evaluate_maps_on_custom_prompt_by_nouns(args)

    # else:
    #     evaluate_maps_on_custom_prompt(args)
