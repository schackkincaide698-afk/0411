"""Our custom pedestrian dataset."""
import copy
import os
import pickle
from functools import reduce
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from pedgen.utils.colors import IMG_MEAN, IMG_STD
from pedgen.utils.rot import (axis_angle_to_matrix, create_2d_grid,
                              create_occupancy_grid, depth_to_3d,
                              matrix_to_rotation_6d)

class CityWalkersDataset(Dataset):
    """Lightning dataset for pedestrian generation."""

    def __init__(
        self,
        label_file: str,
        mode: str,
        data_root: str,
        img_root: str,
        img_dim: List,
        min_timestamp: int,
        use_partial: bool,
        num_timestamp: int,
        depth_root: str,
        semantic_root: str,
        sample_interval: int,
        sample_start_idx: int,
        grid_size: List,
        grid_points: List,
        scene_voxel_points: List,
        scene_token_points: int,
        mode_train_target: str,
        use_image: bool,
        use_data_augmentation: bool,
        train_percent: float,
    ) -> None:
        with open(os.path.join(data_root, label_file), "rb") as f:
            labels = pickle.load(f)#读取WHAM的.pkl文件
        self.label_list = []
        self.img_mean = np.array(IMG_MEAN)
        self.img_std = np.array(IMG_STD)
        self.img_w = img_dim[1]
        self.img_h = img_dim[0]
        self.mode = mode
        self.num_timestamp = num_timestamp
        self.min_timestamp = min_timestamp  # hardcoded
        self.use_partial = use_partial
        self.grid_size = grid_size
        self.use_image = use_image
        self.grid_points = grid_points
        self.scene_voxel_points = scene_voxel_points
        self.scene_token_points = scene_token_points
        self.mode_train_target = mode_train_target
        self.use_data_augmentation = use_data_augmentation

        for idx, val in enumerate(labels):
            if np.isnan(val["global_trans"]).any() or np.isnan(
                    val["local_trans"]).any():
                continue
            image_path = os.path.join(data_root, img_root, val["image"])
            if self.mode != "pred":
                i = sample_start_idx
                max_i = val["global_trans"].shape[
                    0] - self.min_timestamp + 1 if self.use_partial and self.mode == "train" else val[
                        "global_trans"].shape[0] - self.num_timestamp + 1
                while i < max_i:
                    safe_path = image_path.replace('\\', '/')
                    img_id = int(image_path.split("/")[-1].split(".")[0]) + i
                    new_val = copy.deepcopy(val)
                    new_val["start_t"] = i
                    new_image_path = image_path[:-10] + str(img_id).zfill(
                        6) + ".jpg"
                    new_val["image"] = new_image_path
        
                    depth_path = new_image_path.replace(img_root, depth_root)
                    depth_path = depth_path.replace("jpg", "png")
                    new_val["depth"] = depth_path

                    semantic_path = new_image_path.replace(
                        img_root, semantic_root)
                    semantic_path = semantic_path.replace("jpg", "png")
                    new_val["semantic"] = semantic_path

                    points_path = new_image_path.replace(img_root, "points")
                    points_path = points_path.replace("jpg", "npy")
                    new_val["points"] = points_path
                    new_val["index"] = idx

                    img_id = new_val["image"].split("/")
                    img_id = img_id[-2] + "_" + img_id[-1][:-4] + "_" + str(idx)
                    img_id = img_id[-2] + "_" + \
                    img_id[-1][:-4] + "_" + str(idx)
                    self.label_list.append(new_val)
                    i += sample_interval
            else:
                val["start_t"] = 0
                val["image"] = image_path
                if not os.path.exists(val["image"]):
                    continue
                depth_path = image_path.replace(img_root, depth_root)
                depth_path = depth_path.replace("jpg", "png")
                val["depth"] = depth_path
                semantic_path = image_path.replace(img_root, semantic_root)
                semantic_path = semantic_path.replace("jpg", "png")
                val["semantic"] = semantic_path
                points_path = image_path.replace(img_root, "points")
                points_path = points_path.replace("jpg", "npy")
                val["points"] = points_path
                self.label_list.append(val)

        if self.mode == "train" and train_percent < 1.0:
            self.label_list = self.label_list[:int(
                len(self.label_list) * train_percent)]

    def __len__(self) -> int:
        return len(self.label_list)
    
    # 每个点是4维：[x,y,z,semantic]，直接生成固定长度scene_tokens
    def load_scene_tokens(self, label, intrinsics_old):
        tt = lambda x: torch.from_numpy(x).float()

        try:
            depth_3d = np.load(label["points"])
            depth_3d = tt(depth_3d)  # [H, W, 4]
        except Exception:
            depth = cv2.imread(label["depth"], -1)
            depth = np.array(depth, dtype=np.float32) / 256.0
            depth_3d = depth_to_3d(depth, intrinsics_old)
            depth_3d = tt(depth_3d)

            semantic_raw = cv2.imread(label["semantic"], -1)
            semantic_raw = np.array(semantic_raw, dtype=np.float32)
            semantic_raw = tt(semantic_raw).unsqueeze(-1)#增加一维 h,w,1

            depth_3d = torch.cat([depth_3d, semantic_raw], dim=-1)#(X, Y, Z, 语义值)

            points_dir = label["points"].rsplit("/", 1)[0]
            os.makedirs(points_dir, exist_ok=True)
            np.save(label["points"], depth_3d.cpu().numpy())

        points = depth_3d.reshape(-1, 4)

        valid = torch.isfinite(points).all(dim=-1)
        valid = valid & (points[:, 2] > 1e-5)
        points = points[valid]

        #现在的“均匀下采样”只是按一维展开顺序抽点，不是真正空间均匀
        num_target = 4096
        num_points = points.shape[0]

        if num_points == 0:
            # 极小概率的容错：深度图全黑
            return torch.zeros((num_target, 4), dtype=torch.float32)

        if num_points >= num_target:
            # 确定性均匀下采样：按比例步长截取，最大程度保留全局空间分布
            step = num_points / num_target
            indices = (torch.arange(num_target).float() * step).long()
            points = points[indices]
        else:
            # 确定性填充：顺序重复已有序列，不引入额外的随机噪声
            pad_indices = torch.arange(num_target - num_points) % num_points
            points = torch.cat([points, points[pad_indices]], dim=0)

        points = points.clone()
        grid_size = torch.tensor(self.grid_size, dtype=torch.float32)
        voxel_points = torch.tensor(self.scene_voxel_points, dtype=torch.float32)
        voxel_size = torch.tensor([
            (grid_size[1] - grid_size[0]) / voxel_points[0],
            (grid_size[3] - grid_size[2]) / voxel_points[1],
            (grid_size[5] - grid_size[4]) / voxel_points[2],
        ], dtype=torch.float32)
        grid_lower_bound = torch.tensor([grid_size[0], grid_size[2], grid_size[4]], dtype=torch.float32)

        grid_mask = (
            (points[:, 0] >= grid_size[0]) &
            (points[:, 0] < grid_size[1]) &
            (points[:, 1] >= grid_size[2]) &
            (points[:, 1] < grid_size[3]) &
            (points[:, 2] >= grid_size[4]) &
            (points[:, 2] < grid_size[5])
        )
        points = points[grid_mask]
        if points.shape[0] == 0:
            points = torch.zeros((1, 4), dtype=torch.float32)

        indices = ((points[:, :3] - grid_lower_bound.unsqueeze(0)) / voxel_size.unsqueeze(0)).floor().long()
        indices[:, 0] = indices[:, 0].clamp(0, self.scene_voxel_points[0] - 1)
        indices[:, 1] = indices[:, 1].clamp(0, self.scene_voxel_points[1] - 1)
        indices[:, 2] = indices[:, 2].clamp(0, self.scene_voxel_points[2] - 1)
        voxel_hash = (
            indices[:, 0] * (self.scene_voxel_points[1] * self.scene_voxel_points[2]) +
            indices[:, 1] * self.scene_voxel_points[2] +
            indices[:, 2]
        )
        unique_hash, inverse = torch.unique(voxel_hash, sorted=False, return_inverse=True)
        num_voxels = unique_hash.shape[0]

        xyz_sum = torch.zeros((num_voxels, 3), dtype=torch.float32)
        xyz_sum.index_add_(0, inverse, points[:, :3])
        counts = torch.bincount(inverse, minlength=num_voxels).float().unsqueeze(-1).clamp(min=1.0)
        xyz_mean = xyz_sum / counts

        semantic_idx = points[:, 3].long().clamp(min=0, max=18)
        semantic_count = torch.zeros((num_voxels, 19), dtype=torch.float32)
        semantic_count.index_put_(
            (inverse, semantic_idx),
            torch.ones_like(semantic_idx, dtype=torch.float32),
            accumulate=True,
        )
        semantic_mode = torch.argmax(semantic_count, dim=-1).float().unsqueeze(-1)
        scene_tokens = torch.cat([xyz_mean, semantic_mode], dim=-1)

        if scene_tokens.shape[0] > self.scene_token_points:
            distances = torch.norm(scene_tokens[:, :3], dim=-1)
            topk_idx = torch.topk(distances, k=self.scene_token_points, largest=False).indices
            scene_tokens = scene_tokens[topk_idx]
        elif scene_tokens.shape[0] < self.scene_token_points:
            pad_num = self.scene_token_points - scene_tokens.shape[0]
            pad_tokens = scene_tokens[:1].repeat(pad_num, 1)
            scene_tokens = torch.cat([scene_tokens, pad_tokens], dim=0)

        return scene_tokens

    # Diffuser分支使用的原始点云（用于occupancy_grid）
    def load_scene_points_raw(self, label, intrinsics_old):
        tt = lambda x: torch.from_numpy(x).float()
        try:
            depth_3d = np.load(label["points"])
            depth_3d = tt(depth_3d)
        except Exception:
            depth = cv2.imread(label["depth"], -1)
            depth = np.array(depth, dtype=np.float32) / 256.0
            depth_3d = depth_to_3d(depth, intrinsics_old)
            depth_3d = tt(depth_3d)
            semantic_raw = cv2.imread(label["semantic"], -1)
            semantic_raw = np.array(semantic_raw, dtype=np.float32)
            semantic_raw = tt(semantic_raw).unsqueeze(-1)
            depth_3d = torch.cat([depth_3d, semantic_raw], dim=-1)
            points_dir = label["points"].rsplit("/", 1)[0]
            os.makedirs(points_dir, exist_ok=True)
            np.save(label["points"], depth_3d.cpu().numpy())
        return depth_3d
    
    def __getitem__(self, index: int) -> Dict:
        label = self.label_list[index]
        tt = lambda x: torch.from_numpy(x).float()
        data_dict = {}
        img_id = label["image"].split("/")
        img_id = img_id[-2] + "_" + img_id[-1][:-4] + "_" + str(label["index"])
        meta = {"source": "pedmotion", "img_id": img_id}
        data_dict["meta"] = meta

        if self.use_image:
            img = torch.zeros([3, 720, 1280], dtype=torch.float32)
        else:
            rgb = cv2.imread(label["image"])
            rgb = np.array(rgb, dtype=np.float32)
            rgb_vis = copy.deepcopy(rgb)
            mean = np.float64(self.img_mean.reshape(1, -1))
            stdinv = 1 / np.float64(self.img_std.reshape(1, -1))
            cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB, rgb)  # type: ignore
            cv2.subtract(rgb, mean, rgb)  # type: ignore
            cv2.multiply(rgb, stdinv, rgb)  # type: ignore
            img = tt(rgb).permute(2, 0, 1)  # 3, H ,W

        f = (1280**2 + 720**2)**0.5
        cx = 0.5 * 1280
        cy = 0.5 * 720
        intrinsics_old = np.eye(3)
        intrinsics_old[0, 0] = f
        intrinsics_old[1, 1] = f
        intrinsics_old[0, 2] = cx
        intrinsics_old[1, 2] = cy

        start_t = label["start_t"]

        global_orient_source = axis_angle_to_matrix(
            tt(label["global_orient"][start_t])
        )
        global_orient_target = axis_angle_to_matrix(
            tt(label["local_orient"][start_t])
        )

        transl_source = tt(label["global_trans"][[start_t]])
        transl_target = tt(label["local_trans"][[start_t]])

        source_to_target_rotation = global_orient_target @ global_orient_source.T

        global_orient = axis_angle_to_matrix(
            tt(label["global_orient"][start_t:, :3])
        )
        global_orient = source_to_target_rotation @ global_orient

        source_to_target_translation = (
            transl_target.T - source_to_target_rotation @ transl_source.T
        )

        transl = tt(label["global_trans"][start_t:])
        transl = source_to_target_rotation @ transl.T + source_to_target_translation
        transl = transl.T

        data_dict["img"] = img
        data_dict["intrinsics"] = tt(intrinsics_old)
        data_dict["global_trans"] = transl[:self.num_timestamp]
        data_dict["motion_mask"] = torch.zeros((self.num_timestamp,), dtype=torch.bool)

        data_dict["global_orient"] = matrix_to_rotation_6d(
            global_orient[:self.num_timestamp]
        )
        data_dict["betas"] = torch.mean(
            tt(label["betas"][start_t:start_t + self.num_timestamp]),
            dim=0
        )

        data_dict["body_pose"] = matrix_to_rotation_6d(
            axis_angle_to_matrix(
                tt(label["body_pose"][start_t:start_t + self.num_timestamp])
            )
        ).reshape(-1, 23 * 6)

        if data_dict["global_trans"].shape[0] < self.num_timestamp:
            motion_length = data_dict["global_trans"].shape[0]
            data_dict["motion_mask"][motion_length:] = True
            data_dict["global_trans"] = torch.cat([
                data_dict["global_trans"],
                torch.zeros(self.num_timestamp - motion_length, 3)
            ], dim=0)
            data_dict["global_orient"] = torch.cat([
                data_dict["global_orient"],
                torch.zeros(self.num_timestamp - motion_length, 6)
            ], dim=0)
            data_dict["body_pose"] = torch.cat([
                data_dict["body_pose"],
                torch.zeros(self.num_timestamp - motion_length, 23 * 6)
            ], dim=0)

        if not self.use_image:
            return data_dict
        
        if not self.mode == "pred":#不是推理模式
            # Predictor条件构造
            valid_len = self.num_timestamp - int(data_dict["motion_mask"].sum().item())
            valid_len = max(valid_len, 1)
            data_dict["gt_goal_rel"] = (
                data_dict["global_trans"][valid_len - 1] - data_dict["global_trans"][0]
            ).clone()

            # Global planner supervision with non-overlapping horizon:
            # [0~60], [60~120], [120~149].
            full_horizon = 150
            traj_full = transl[:full_horizon]
            if traj_full.shape[0] < full_horizon:
                traj_full = torch.cat([
                    traj_full,
                    torch.zeros(full_horizon - traj_full.shape[0], 3)
                ], dim=0)
            valid_horizon = min(transl.shape[0], full_horizon)
            last_valid = max(valid_horizon - 1, 0)
            seg_pairs = [(0, 60), (60, 120), (120, 149)]
            init_anchor_idx = [0, 60, 120]
            goal_seq = []
            valid_seq_mask = []
            init_seq = []
            init_seq_mask = []
            for start_idx in init_anchor_idx:
                s = min(start_idx, last_valid)
                init_seq.append(traj_full[s])
                init_seq_mask.append(1.0 if last_valid >= start_idx else 0.0)
            for start_idx, end_idx in seg_pairs:
                s = min(start_idx, last_valid)
                e = min(end_idx, last_valid)
                goal_seq.append(traj_full[e] - traj_full[s])
                valid_seq_mask.append(1.0 if last_valid >= end_idx else 0.0)
            data_dict["gt_init_pos_seq"] = torch.stack(init_seq, dim=0)
            data_dict["gt_init_pos_seq_mask"] = torch.tensor(init_seq_mask, dtype=torch.float32)
            data_dict["gt_goal_rel_seq"] = torch.stack(goal_seq, dim=0)
            data_dict["gt_goal_rel_seq_mask"] = torch.tensor(valid_seq_mask, dtype=torch.float32)
            data_dict["gt_init_pos"] = data_dict["gt_init_pos_seq"][0].clone()
            data_dict["gt_goal_rel"] = data_dict["gt_goal_rel_seq"][0].clone()
            data_dict["gt_traj_150"] = traj_full.clone()
            data_dict["gt_traj_150_mask"] = (torch.arange(full_horizon) <= last_valid).float()
            data_dict["grid_size"] = torch.tensor(self.grid_size).float()
            data_dict["grid_points"] = torch.tensor(self.grid_points).long()
            data_dict["scene_tokens"] = self.load_scene_tokens(label, intrinsics_old)

            # Diffuser条件构造（原代码风格）
            if self.use_image:
                depth_3d = self.load_scene_points_raw(label, intrinsics_old).clone()
                init_trans = transl[0]
                dx = int(init_trans[0] * (1280**2 + 720**2)**0.5 / init_trans[2] + 1280 / 2)
                dy = int(init_trans[1] * (1280**2 + 720**2)**0.5 / init_trans[2] + 720 / 2)
                dx = max(min(dx, 1279), 0)
                dy = max(min(dy, 719), 0)
                metric_depth = depth_3d[dy, dx, 2]
                depth_3d[..., :3] = depth_3d[..., :3] * init_trans[2] / metric_depth
                depth_3d[..., :3] = depth_3d[..., :3] - init_trans.unsqueeze(0).unsqueeze(0)

                # 仅Diffuser分支做数据增强
                if self.mode == "train" and self.use_data_augmentation:
                    transl = transl - transl[[0]]
                    rot = np.random.uniform(-np.pi, np.pi)
                    rot_mat = torch.tensor(
                        [[np.cos(rot), 0, np.sin(rot)],
                         [0, 1, 0],
                         [-np.sin(rot), 0, np.cos(rot)]],
                        dtype=global_orient.dtype,
                    )
                    global_orient = rot_mat @ global_orient
                    transl = transl @ rot_mat.T
                    depth_3d[..., :3] = depth_3d[..., :3] @ rot_mat.T

                mask = torch.ones([720, 1280], dtype=torch.bool)
                bbox = label["bbox_2d"][start_t]
                mask[int(bbox[2]):int(bbox[3]), int(bbox[0]):int(bbox[1])] = 0
                mask = reduce(torch.logical_and, [
                    mask,
                    depth_3d[..., 0] >= self.grid_size[0] + 1e-5,
                    depth_3d[..., 0] < self.grid_size[1] - 1e-5,
                    depth_3d[..., 1] >= self.grid_size[2] + 1e-5,
                    depth_3d[..., 1] < self.grid_size[3] - 1e-5,
                    depth_3d[..., 2] >= self.grid_size[4] + 1e-5,
                    depth_3d[..., 2] < self.grid_size[5] - 1e-5,
                ])
                depth_3d = depth_3d.reshape(720 * 1280, -1)
                depth_3d = depth_3d[mask.flatten(), :]
                occupancy_grid = create_occupancy_grid(depth_3d, self.grid_size, self.grid_points)
                grid_2d = tt(create_2d_grid(num_points=self.grid_points, grid_size=self.grid_size))
                occupancy_grid = occupancy_grid.permute(0, 2, 1)
                occupancy_grid = torch.cat([occupancy_grid, grid_2d], dim=-1)
                occupancy_grid = occupancy_grid.reshape(occupancy_grid.shape[0] * occupancy_grid.shape[1], -1)
                data_dict["new_img"] = occupancy_grid
        else:
            # 推理模式：原代码风格，pred_goal改为3段
            pred_goal = torch.zeros((3, 3), dtype=torch.float32)
            num_pred_steps = pred_goal.shape[0]
            data_dict["img"] = data_dict["img"].unsqueeze(0).repeat(num_pred_steps, 1, 1, 1)
            data_dict["global_trans"] = data_dict["global_trans"].unsqueeze(0).repeat(num_pred_steps, 1, 1)
            data_dict["global_orient"] = data_dict["global_orient"].unsqueeze(0).repeat(num_pred_steps, 1, 1)
            data_dict["betas"] = data_dict["betas"].unsqueeze(0).repeat(num_pred_steps, 1)
            data_dict["body_pose"] = data_dict["body_pose"].unsqueeze(0).repeat(num_pred_steps, 1, 1)
            data_dict["gt_init_pos"] = data_dict["gt_init_pos"].unsqueeze(0).repeat(num_pred_steps, 1)
            data_dict["batch_size"] = num_pred_steps
            data_dict["grid_size"] = torch.tensor(self.grid_size).float()
            data_dict["grid_points"] = torch.tensor(self.grid_points).long()
            data_dict["scene_tokens"] = self.load_scene_tokens(label, intrinsics_old)

        return data_dict


def collate_fn_pedmotion_predictor(data):
    img_batch = []
    intrinsics_batch = []
    global_trans_batch = []
    betas_batch = []
    global_orient_batch = []
    body_pose_batch = []
    meta_batch = []
    motion_mask_batch = []
    gt_init_pos_batch = []
    gt_goal_rel_batch = []
    gt_init_pos_seq_batch = []
    gt_init_pos_seq_mask_batch = []
    gt_goal_rel_seq_batch = []
    gt_goal_rel_seq_mask_batch = []
    gt_traj_150_batch = []
    gt_traj_150_mask_batch = []
    scene_tokens_batch = []

    for data_dict in data:
        img_batch.append(data_dict["img"])
        intrinsics_batch.append(data_dict["intrinsics"])
        global_trans_batch.append(data_dict["global_trans"])
        betas_batch.append(data_dict["betas"])
        global_orient_batch.append(data_dict["global_orient"])
        body_pose_batch.append(data_dict["body_pose"])
        meta_batch.append(data_dict["meta"])

        if "motion_mask" in data_dict:
            motion_mask_batch.append(data_dict["motion_mask"])
        if "gt_init_pos" in data_dict:
            gt_init_pos_batch.append(data_dict["gt_init_pos"])
        if "gt_goal_rel" in data_dict:
            gt_goal_rel_batch.append(data_dict["gt_goal_rel"])
        if "gt_init_pos_seq" in data_dict:
            gt_init_pos_seq_batch.append(data_dict["gt_init_pos_seq"])
        if "gt_init_pos_seq_mask" in data_dict:
            gt_init_pos_seq_mask_batch.append(data_dict["gt_init_pos_seq_mask"])
        if "gt_goal_rel_seq" in data_dict:
            gt_goal_rel_seq_batch.append(data_dict["gt_goal_rel_seq"])
        if "gt_goal_rel_seq_mask" in data_dict:
            gt_goal_rel_seq_mask_batch.append(data_dict["gt_goal_rel_seq_mask"])
        if "gt_traj_150" in data_dict:
            gt_traj_150_batch.append(data_dict["gt_traj_150"])
        if "gt_traj_150_mask" in data_dict:
            gt_traj_150_mask_batch.append(data_dict["gt_traj_150_mask"])
        if "scene_tokens" in data_dict:
            scene_tokens_batch.append(data_dict["scene_tokens"])


    ret_dict = {
        "img": torch.stack(img_batch),
        "intrinsics": torch.stack(intrinsics_batch),
        "meta": meta_batch,
        "global_trans": torch.stack(global_trans_batch),
        "betas": torch.stack(betas_batch),
        "global_orient": torch.stack(global_orient_batch),
        "body_pose": torch.stack(body_pose_batch),
        "batch_size": len(img_batch),
        "scene_tokens": torch.stack(scene_tokens_batch),
        "grid_size": data[0]["grid_size"],
        "grid_points": data[0]["grid_points"],
    }
    if len(gt_init_pos_batch) > 0:
        ret_dict["gt_init_pos"] = torch.stack(gt_init_pos_batch)
    if len(gt_goal_rel_batch) > 0:
        ret_dict["gt_goal_rel"] = torch.stack(gt_goal_rel_batch)
    if len(gt_init_pos_seq_batch) > 0:
        ret_dict["gt_init_pos_seq"] = torch.stack(gt_init_pos_seq_batch)
    if len(gt_init_pos_seq_mask_batch) > 0:
        ret_dict["gt_init_pos_seq_mask"] = torch.stack(gt_init_pos_seq_mask_batch)
    if len(gt_goal_rel_seq_batch) > 0:
        ret_dict["gt_goal_rel_seq"] = torch.stack(gt_goal_rel_seq_batch)
    if len(gt_goal_rel_seq_mask_batch) > 0:
        ret_dict["gt_goal_rel_seq_mask"] = torch.stack(gt_goal_rel_seq_mask_batch)
    if len(gt_traj_150_batch) > 0:
        ret_dict["gt_traj_150"] = torch.stack(gt_traj_150_batch)
    if len(gt_traj_150_mask_batch) > 0:
        ret_dict["gt_traj_150_mask"] = torch.stack(gt_traj_150_mask_batch)

    if len(motion_mask_batch) > 0:
        ret_dict["motion_mask"] = torch.stack(motion_mask_batch)

    return ret_dict

def collate_fn_pedmotion_diffuser(data):
    img_batch = []
    intrinsics_batch = []
    global_trans_batch = []
    betas_batch = []
    global_orient_batch = []
    body_pose_batch = []
    meta_batch = []
    motion_mask_batch = []
    new_img_batch = []

    for data_dict in data:
        img_batch.append(data_dict["img"])
        intrinsics_batch.append(data_dict["intrinsics"])
        global_trans_batch.append(data_dict["global_trans"])
        betas_batch.append(data_dict["betas"])
        global_orient_batch.append(data_dict["global_orient"])
        body_pose_batch.append(data_dict["body_pose"])
        meta_batch.append(data_dict["meta"])
        if "motion_mask" in data_dict:
            motion_mask_batch.append(data_dict["motion_mask"])
        if "new_img" in data_dict:
            new_img_batch.append(data_dict["new_img"])

    ret_dict = {
        "img": torch.stack(img_batch),
        "intrinsics": torch.stack(intrinsics_batch),
        "meta": meta_batch,
        "global_trans": torch.stack(global_trans_batch),
        "betas": torch.stack(betas_batch),
        "global_orient": torch.stack(global_orient_batch),
        "body_pose": torch.stack(body_pose_batch),
        "batch_size": len(img_batch),
    }
    if len(motion_mask_batch) > 0:
        ret_dict["motion_mask"] = torch.stack(motion_mask_batch)
    if len(new_img_batch) > 0:
        ret_dict["new_img"] = torch.stack(new_img_batch)
    return ret_dict

def collate_fn_pedmotion(data):
    if len(data) > 0 and "new_img" in data[0]:
        return collate_fn_pedmotion_diffuser(data)
    return collate_fn_pedmotion_predictor(data)

def collate_fn_pedmotion_pred(data):
    """Custom collate function."""
    return collate_fn_pedmotion_predictor(data)
