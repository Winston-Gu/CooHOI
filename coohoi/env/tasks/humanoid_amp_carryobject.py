import os
import json
import torch
import random
from tqdm import tqdm
import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *

import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils


class HumanoidAMPCarryObject(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.num_envs = cfg["env"]["numEnvs"]

        self._box_dist_min = 0.5
        self._box_dist_max = 10
        self._target_dist_min = 1.5
        self._target_dist_max = 10

        # scaling object size
        self._box_min_scale = 1.0
        self._box_max_scale = 1.0
        self.scaling_factor = self._box_min_scale + \
            (self._box_max_scale - self._box_min_scale) * \
            torch.rand(self.num_envs)

        # scaling object weight
        self._box_min_weight = 0.8
        self._box_max_weight = 0.8
        self.scaling_factor_weight = self._box_min_weight + \
            (self._box_max_weight - self._box_min_weight) * \
            torch.rand(self.num_envs)

        self._default_box_width_size = 0.5
        self._default_box_length_size = 0.5
        self._default_box_height_size = 0.5

        self.obs_add_noise = True
        self.noise_level = 0.1

        device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self._width_box_size = torch.zeros(self.num_envs).to(device)
        self._length_box_size = torch.zeros(self.num_envs).to(device)
        self._height_box_size = torch.zeros(self.num_envs).to(device)

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self.spacing = cfg["env"]['envSpacing']
        self.reset_time = 0

        self.log_success = False

        if cfg['env']['eval_mode']:
            # for calculate success rate and distance error and execution time
            self.log_success = True
            self._distance_to_target = [
                [] for _ in range(self.num_envs)]
            self.log_success_rate = []
            self.log_success_precision = []

        if cfg['env']['save_motions']:
            self.save_motion_for_blender = False
            self._save_all_state = True
            self.record_frame_number = 600

            self.output_dict = {}
            self.output_dict['trans'] = np.zeros(
                [self.record_frame_number, 15, 3])
            self.output_dict['rot'] = np.zeros(
                [self.record_frame_number, 15, 4])
            self.output_dict['obj_pos'] = np.zeros(
                [self.record_frame_number, 3])
            self.output_dict['obj_rot'] = np.zeros(
                [self.record_frame_number, 4])

            if self._save_all_state:
                self.output_dict['root_pos'] = np.zeros(
                    [self.record_frame_number, 3])
                self.output_dict['root_rot'] = np.zeros(
                    [self.record_frame_number, 4])
                self.output_dict['dof_pos'] = np.zeros(
                    [self.record_frame_number, 28])
        self.record_step = 0

        width_half_size = self._width_box_size / 2.0
        length_half_size = self._length_box_size / 2.0
        height_half_size = self._height_box_size / 2.0

        lfus = torch.stack(
            [-length_half_size, width_half_size, height_half_size], dim=1)
        lfds = torch.stack(
            [-length_half_size, width_half_size, -height_half_size], dim=1)
        lbus = torch.stack(
            [-length_half_size, -width_half_size, height_half_size], dim=1)
        lbds = torch.stack(
            [-length_half_size, -width_half_size, -height_half_size], dim=1)
        rfus = torch.stack(
            [length_half_size, width_half_size, height_half_size], dim=1)
        rfds = torch.stack(
            [length_half_size, width_half_size, -height_half_size], dim=1)
        rbus = torch.stack(
            [length_half_size, -width_half_size, height_half_size], dim=1)
        rbds = torch.stack(
            [length_half_size, -width_half_size, -height_half_size], dim=1)

        stand_points_left = torch.stack(
            [-length_half_size - 0.2, torch.zeros(self.num_envs).to(device), torch.zeros(self.num_envs).to(device)], dim=1)
        stand_points_right = torch.stack(
            [length_half_size + 0.2, torch.zeros(self.num_envs).to(device), torch.zeros(self.num_envs).to(device)], dim=1)
        held_points_left = torch.stack(
            [-length_half_size + width_half_size, torch.zeros(self.num_envs).to(device), torch.zeros(self.num_envs).to(device)], dim=1)
        held_points_right = torch.stack(
            [length_half_size - width_half_size, torch.zeros(self.num_envs).to(device), torch.zeros(self.num_envs).to(device)], dim=1)

        self.box_bps = torch.stack(
            [lfus, lfds, lbus, lbds, rfus, rfds, rbus, rbds], dim=0)

        self.stand_held_points_offset = torch.stack(
            [stand_points_left, stand_points_right, held_points_left, held_points_right], dim=0)

        self._prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_box_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float)

        lift_body_names = cfg["env"]["liftBodyNames"]
        self._lift_body_ids = self._build_lift_body_ids_tensor(lift_body_names)

        self._build_box_tensors()
        # self._build_platform_tensors()
        self._build_target_state_tensors()

    def _build_lift_body_ids_tensor(self, lift_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in lift_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name)
            assert (body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._box_handles = []
        self._load_box_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_box_asset(self):
        width_box_size = self._default_box_width_size
        length_box_size = self._default_box_length_size
        height_box_size = self._default_box_height_size

        self._box_asset = []
        self.asset_density = torch.zeros(self.num_envs).to(self.device)

        for env_id in range(self.num_envs):
            scaling_factor_l = self.scaling_factor[env_id]
            scaling_factor_w = self.scaling_factor[env_id]
            scaling_factor_h = self.scaling_factor[env_id]
            scaling_factor_weight = self.scaling_factor_weight[env_id]

            box_length = scaling_factor_l * length_box_size
            box_width = scaling_factor_w * width_box_size
            box_height = scaling_factor_h * height_box_size

            asset_options = gymapi.AssetOptions()
            asset_options.density = scaling_factor_weight * 50.0 / \
                (scaling_factor_l * scaling_factor_w * scaling_factor_h)
            self.asset_density[env_id] = asset_options.density

            self._box_asset.append(self.gym.create_box(
                self.sim, box_length, box_width, box_height, asset_options))

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_box(env_id, env_ptr)
        return

    def _build_box(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 3.0

        scaling_factor_l = self.scaling_factor[env_id]
        scaling_factor_w = self.scaling_factor[env_id]
        scaling_factor_h = self.scaling_factor[env_id]

        self._width_box_size[env_id] = scaling_factor_w * \
            self._default_box_width_size
        self._length_box_size[env_id] = scaling_factor_l * \
            self._default_box_length_size
        self._height_box_size[env_id] = scaling_factor_h * \
            self._default_box_height_size

        box_handle = self.gym.create_actor(
            env_ptr, self._box_asset[env_id], default_pose, "box", col_group, col_filter, segmentation_id)
        props = self.gym.get_actor_dof_properties(env_ptr, box_handle)
        props['friction'].fill(5.0)
        self.gym.set_actor_dof_properties(env_ptr, box_handle, props)
        self._box_handles.append(box_handle)
        return

    def _build_target_state_tensors(self):
        self._target_pos = torch.zeros(self.num_envs, 3).to(self.device)
        self._target_rot = torch.zeros(self.num_envs, 4).to(self.device)
        self.tar_standing_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self.tar_held_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        return

    def _build_box_tensors(self):
        num_actors = self.get_num_actors_per_env()

        self._box_states = self._root_states.view(
            self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        self.box_standing_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self.box_held_points = torch.zeros(
            self.num_envs, 3).to(self.device)
        self._box_actor_ids = to_torch(
            num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        self._box_pos = self._box_states[..., :3]
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(
            self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._box_contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)

        if self.headless:
            self._reset_box(env_ids, randomize=True)
            self._reset_target(env_ids, randomize=True)
        else:
            self._reset_box(env_ids, randomize=False)
            self._reset_target(env_ids, randomize=False)
        # self._reset_platform(env_ids, randomize=True)
        return

    def _reset_box(self, env_ids, randomize=True):

        n = len(env_ids)

        if randomize:
            rand_dist = (self._box_dist_max - self._box_dist_min) * torch.rand(
                [n], dtype=self._box_states.dtype, device=self._box_states.device) + self._box_dist_min
            random_numbers = torch.rand(
                [n], dtype=self._box_states.dtype, device=self._box_states.device)
            random_numbers_rot = torch.rand(
                [n], dtype=self._box_states.dtype, device=self._box_states.device)

            rand_theta = 0 * np.pi * random_numbers
            # rand_theta = (random_numbers - 0.5) * np.pi / 2.0
            self._box_states[env_ids, 0] = rand_dist * \
                torch.cos(rand_theta) + self._humanoid_root_states[env_ids, 0]
            self._box_states[env_ids, 1] = rand_dist * \
                torch.sin(rand_theta) + self._humanoid_root_states[env_ids, 1]
            self._box_states[env_ids, 2] = self._height_box_size[env_ids] / 2.0

            rand_rot_theta = 2 * np.pi * random_numbers_rot
            axis = torch.tensor(
                [0.0, 0.0, 1.0], dtype=self._box_states.dtype, device=self._box_states.device)
            rand_rot = quat_from_angle_axis(rand_theta, axis)
            self._box_states[env_ids, 3:7] = rand_rot
            self._box_states[env_ids, 7:] = 0.0

        else:
            self._box_states[env_ids,
                             0] = self._humanoid_root_states[env_ids, 0] + 3.0
            self._box_states[env_ids,
                             1] = self._humanoid_root_states[env_ids, 1]
            self._box_states[env_ids, 2] = self._height_box_size[env_ids] / 2.0
            self._box_states[env_ids, 3:7] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=self._box_states.dtype, device=self._box_states.device)
            self._box_states[env_ids, 7:] = 0.0
        return

    def _reset_target(self, env_ids, randomize=True):

        n = len(env_ids)
        if randomize:
            rand_dist = (self._target_dist_max - self._target_dist_min) * torch.rand(
                [n], dtype=self._target_pos.dtype, device=self._target_pos.device) + self._target_dist_min
            random_numbers = torch.rand(
                [n], dtype=self._target_pos.dtype, device=self._target_pos.device)
            random_numbers_rot = torch.rand(
                [n], dtype=self._target_pos.dtype, device=self._target_pos.device)

            rand_theta = 0 * np.pi * random_numbers
            self._target_pos[env_ids, 0] = rand_dist * \
                torch.cos(rand_theta) + self._box_states[env_ids, 0]
            self._target_pos[env_ids, 1] = rand_dist * \
                torch.sin(rand_theta) + self._box_states[env_ids, 1]
            self._target_pos[env_ids, 2] = self._height_box_size[env_ids] / 2.0
            
            rand_rot_theta = 2 * np.pi * random_numbers_rot
            axis = torch.tensor(
                [0.0, 0.0, 1.0], dtype=self._target_pos.dtype, device=self._target_pos.device)
            rand_rot = quat_from_angle_axis(rand_theta, axis)
            self._target_rot[env_ids] = rand_rot
            # self._new_marker_states = self._marker_states.clone()

        else:
            self._target_pos[env_ids,
                             0] = self._box_states[env_ids, 0] + 0.0
            self._target_pos[env_ids,
                             1] = self._box_states[env_ids, 1] + 3.0
            self._target_pos[env_ids, 2] = self._height_box_size[env_ids] / 2.0
            self._target_rot[env_ids] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0], dtype=self._target_pos.dtype, device=self._target_pos.device)
        return

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)
        box_env_ids_int32 = self._box_actor_ids[env_ids]
        reset_env_ids_int32 = box_env_ids_int32

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(reset_env_ids_int32), len(reset_env_ids_int32))
        return

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_box_pos[:] = self._box_states[..., 0:3]
        return

    def update_standing_and_held_points(self, box_states, tar_pos, tar_rot, env_ids=None):
        if env_ids is None:
            box_pos = box_states[..., 0:3]
            box_rot = box_states[..., 3:7]
            # hard code: set standing points to the left
            self.box_standing_points[:] = box_pos + \
                quat_rotate(box_rot, self.stand_held_points_offset[0])
            self.box_standing_points[..., 2] = 0.0
            self.box_held_points[:] = box_pos + \
                quat_rotate(box_rot, self.stand_held_points_offset[2])
            # hard code: set standing points to the left
            self.tar_standing_points[:] = tar_pos + \
                quat_rotate(tar_rot, self.stand_held_points_offset[0])
            self.tar_held_points[:] = tar_pos + \
                quat_rotate(tar_rot, self.stand_held_points_offset[2])

        else:
            box_pos = box_states[env_ids, 0:3]
            box_rot = box_states[env_ids, 3:7]
            tar_pos = tar_pos[env_ids]
            tar_rot = tar_rot[env_ids]
            self.box_standing_points[env_ids] = box_pos + \
                quat_rotate(box_rot,
                            self.stand_held_points_offset[0][env_ids])
            self.box_standing_points[env_ids, 2] = 0.0
            self.box_held_points[env_ids] = box_pos + \
                quat_rotate(box_rot,
                            self.stand_held_points_offset[2][env_ids])
            self.tar_standing_points[env_ids] = tar_pos + \
                quat_rotate(tar_rot, self.stand_held_points_offset[0][env_ids])
            self.tar_held_points[env_ids] = tar_pos + \
                quat_rotate(tar_rot, self.stand_held_points_offset[2][env_ids])
        return

    def _compute_task_obs(self, env_ids=None):
        # device = torch.device(
        #     "cuda") if torch.cuda.is_available() else torch.device("cpu")
        if env_ids is None:
            root_states = self._humanoid_root_states
            box_states = self._box_states
            box_bps = self.box_bps
            tar_pos = self._target_pos
            tar_rot = self._target_rot
            # Note: Update Standing points and held points only can be after the box_states is updated
            self.update_standing_and_held_points(box_states, tar_pos, tar_rot)
            box_standing_points = self.box_standing_points
            tar_standing_points = self.tar_standing_points
            density = self.asset_density
        else:
            root_states = self._humanoid_root_states[env_ids]
            box_states = self._box_states[env_ids]
            tar_pos = self._target_pos[env_ids]
            tar_rot = self._target_rot[env_ids]
            box_bps = self.box_bps[:, env_ids, :]
            self.update_standing_and_held_points(
                self._box_states, self._target_pos, self._target_rot, env_ids)
            box_standing_points = self.box_standing_points[env_ids]
            tar_standing_points = self.tar_standing_points[env_ids]
            density = self.asset_density[env_ids]

        obs = compute_carrybox_observations(
            root_states, box_states, tar_pos, tar_rot, box_bps, box_standing_points, tar_standing_points, density
        )

        if self.obs_add_noise:
            # add normal noise to the observation
            num_env = obs.shape[0]
            # add noise on first 39 dimensions
            noise = torch.normal(mean=0.0, std=self.noise_level, size=(
                num_env, 39), device=obs.device, dtype=obs.dtype)
            obs[:, :39] += noise
        if self.cfg['env']['save_motions']:
            if self.record_step < self.record_frame_number:
                self.output_dict['trans'][self.record_step] = self._rigid_body_pos[0].cpu(
                ).numpy()
                self.output_dict['rot'][self.record_step] = self._rigid_body_rot[0].cpu(
                ).numpy()
                self.output_dict['obj_pos'][self.record_step] = box_states[0, 0:3].cpu(
                ).numpy()
                self.output_dict['obj_rot'][self.record_step] = box_states[0, 3:7].cpu(
                ).numpy()
            if self.record_step == self.record_frame_number:
                np.save("motion_sequence/single_carry_box.npy",
                        self.output_dict)
        self.record_step += 1
        if self.log_success:
            distance_to_target = torch.norm(box_states[..., 0:3] - tar_pos, dim=-1)
            self._distance_to_target = np.array(distance_to_target.cpu().numpy())
            success_env_mask = self._distance_to_target< 0.2
            success_env_id = np.where(success_env_mask)
            # success_step = [np.where(self._distance_to_target[i] < 0.2)[0][0] for i in success_env_id]
            # print("Success rate: ", success_env_mask.sum() / self.num_envs)
            # print("Success step: ", np.mean(success_step) / 30)
            if self._distance_to_target[success_env_mask].size > 0:
                mean_error = self._distance_to_target[success_env_mask].mean()
            else:
                mean_error = 100
            # print("mean distance: ", mean_error)
            self.log_success_rate.append(success_env_mask.sum() / self.num_envs)
            self.log_success_precision.append(mean_error)
            print("Max Success rate: ", max(self.log_success_rate))
            print("Max Success precision: ", min(self.log_success_precision))

            print(self.record_step)
        return obs

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            # the original was 75, add 1 dimension for the density of the box
            obs_size = 76
        return obs_size

    def _compute_reset(self):
        box_pos = self._box_states[..., 0:3]
        prev_box_pos = self._prev_box_pos
        dt_tensor = torch.tensor(self.dt, dtype=torch.float32)
        hand_positions = self._rigid_body_pos[..., self._lift_body_ids, :]
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, self.progress_buf, self._contact_forces,
            self._contact_body_ids, self._rigid_body_pos, self._box_contact_forces,
            self._lift_body_ids, self.max_episode_length,
            self._enable_early_termination, self._termination_heights,
            box_pos, prev_box_pos, dt_tensor, hand_positions
        )
        return

    def _compute_reward(self, actions):
        walk_pos_reward_w = 0.1
        walk_vel_reward_w = 0.1
        walk_face_reward_w = 0.1
        held_hand_reward_w = 0.4
        held_height_reward_w = 0.0
        carry_box_reward_pos_far_w = 0.1
        carry_box_reward_velocity_w = 0.0
        carry_box_reward_pos_near_w = 0.2
        carry_box_face_reward_w = 0.2
        carry_box_dir_reward_w = 0.1
        putdown_reward_w = 0.1

        box_pos = self._box_states[..., 0:3]  # Box position
        box_height = box_pos[..., 2]
        box_rot = self._box_states[..., 3:7]  # Box rotation
        prev_box_pos = self._prev_box_pos
        box_standing_pos = self.box_standing_points
        box_held_pos = self.box_held_points
        held_point_height = box_held_pos[..., 2]
        dt_tensor = torch.tensor(self.dt, dtype=torch.float32)

        root_pos = self._humanoid_root_states[..., 0:3]  # 3d state
        root_rot = self._humanoid_root_states[..., 3:7]  # 4d state
        prev_root_pos = self._prev_root_pos
        hand_positions = self._rigid_body_pos[..., self._lift_body_ids, :]
        # box_states = self._box_states[..., 0:3]
        tar_pos = self._target_pos
        tar_rot = self._target_rot

        walk_pos_reward, walk_vel_reward, walk_face_reward = compute_walk_reward(
            root_pos, root_rot, prev_root_pos, box_standing_pos, dt_tensor)
        held_hand_reward = compute_contact_reward(
            hand_positions, box_held_pos, root_pos, box_standing_pos, box_pos, tar_pos)
        height_reward = compute_height_reward(held_point_height)
        carry_box_reward_pos_far, carry_box_reward_velocity, \
            carry_box_reward_pos_near, carry_box_face_reward, \
            carry_box_dir_reward, put_down_height_reward = compute_carry_reward(
                root_pos, root_rot, box_pos, box_rot, prev_box_pos, tar_pos, tar_rot, held_point_height, dt_tensor)

        self.rew_buf[:] = walk_pos_reward_w * walk_pos_reward + \
            walk_vel_reward_w * walk_vel_reward + \
            walk_face_reward_w * walk_face_reward + \
            held_hand_reward_w * held_hand_reward + \
            held_height_reward_w * height_reward + \
            carry_box_reward_pos_far_w * carry_box_reward_pos_far + \
            carry_box_reward_velocity_w * carry_box_reward_velocity + \
            carry_box_reward_pos_near_w * carry_box_reward_pos_near + \
            carry_box_face_reward_w * carry_box_face_reward + \
            carry_box_dir_reward_w * carry_box_dir_reward + \
            putdown_reward_w * put_down_height_reward

        walk_reward = walk_pos_reward_w * walk_pos_reward + \
            walk_vel_reward_w * walk_vel_reward + \
            walk_face_reward_w * walk_face_reward
        contact_reward = held_hand_reward_w * held_hand_reward
        carry_reward = carry_box_reward_pos_far_w * carry_box_reward_pos_far + \
            carry_box_reward_velocity_w * carry_box_reward_velocity + \
            carry_box_reward_pos_near_w * carry_box_reward_pos_near + \
            carry_box_face_reward_w * carry_box_face_reward + \
            carry_box_dir_reward_w * carry_box_dir_reward + \
            putdown_reward_w * put_down_height_reward

        box_half_height = self._height_box_size / 2.0
        height_diff = compute_box_raise_height(box_half_height, box_height)
        return

    def _update_task(self):
        # if self._enable_task_update:
        #     change_steps = torch.randint(low=self._task_finish_steps_min, high=self._task_finish_steps_max, size=(
        #         self.num_envs,), device=self.device, dtype=torch.int64)
        #     reset_task_mask = self._task_finish_steps > change_steps
        #     rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()

        #     if len(rest_env_ids) > 0:
        #         self._reset_task(rest_env_ids)
        return

    def _reset_task(self, env_ids):
        # self._reset_target(env_ids, randomize=True)
        # self._task_finish_steps[env_ids] = 0
        return

    def _draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        tar_pos = self._target_pos
        tar_rot = self._target_rot

        box_bps = self.box_bps
        lfus = box_bps[0]
        lfds = box_bps[1]
        lbus = box_bps[2]
        lbds = box_bps[3]

        rfus = box_bps[4]
        rfds = box_bps[5]
        rbus = box_bps[6]
        rbds = box_bps[7]

        tar_lfus = convert_static_point_to_world(lfus, tar_pos, tar_rot)
        tar_lfds = convert_static_point_to_world(lfds, tar_pos, tar_rot)
        tar_lbus = convert_static_point_to_world(lbus, tar_pos, tar_rot)
        tar_lbds = convert_static_point_to_world(lbds, tar_pos, tar_rot)
        tar_rfus = convert_static_point_to_world(rfus, tar_pos, tar_rot)
        tar_rfds = convert_static_point_to_world(rfds, tar_pos, tar_rot)
        tar_rbus = convert_static_point_to_world(rbus, tar_pos, tar_rot)
        tar_rbds = convert_static_point_to_world(rbds, tar_pos, tar_rot)

        verts1 = torch.cat([tar_lfus, tar_lfds], dim=-1).cpu().numpy()
        verts2 = torch.cat([tar_lbus, tar_lbds], dim=-1).cpu().numpy()
        verts3 = torch.cat([tar_rfus, tar_rfds], dim=-1).cpu().numpy()
        verts4 = torch.cat([tar_rbus, tar_rbds], dim=-1).cpu().numpy()
        verts5 = torch.cat([tar_lfus, tar_lbus], dim=-1).cpu().numpy()
        verts6 = torch.cat([tar_lbus, tar_rbus], dim=-1).cpu().numpy()
        verts7 = torch.cat([tar_rbus, tar_rfus], dim=-1).cpu().numpy()
        verts8 = torch.cat([tar_rfus, tar_lfus], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts1[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts2[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts3[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts4[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts5[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts6[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts7[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)
            curr_verts = verts8[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr,
                               curr_verts.shape[0], curr_verts, cols)

        return



#####################################################################
### =========================jit functions=========================###
#####################################################################

@torch.jit.script
def convert_static_point_to_local_observation(point_pos, root_states, central_pos, central_rot):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    point_states = torch.zeros_like(root_states[..., 0:3])
    point_states[:] = point_pos
    rotate_point_staets = quat_rotate(central_rot, point_states)
    target_point_staets = central_pos + rotate_point_staets
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    local_point_pos = quat_rotate(heading_rot, target_point_staets - root_pos)
    return local_point_pos


@torch.jit.script
def convert_static_point_to_world(point_pos, central_pos, central_rot):
    point_states = torch.zeros_like(central_pos[..., 0:3])
    point_states[:] = point_pos
    rotate_point_staets = quat_rotate(central_rot, point_states)
    target_point_staets = central_pos + rotate_point_staets
    return target_point_staets


@torch.jit.script
def compute_carrybox_observations(root_states, box_states, tar_pos, tar_rot, box_bps, box_standing_points, tar_standing_points, density):
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)  # (num_envs, 4)

    box_pos = box_states[:, 0:3]
    box_rot = box_states[:, 3:7]
    box_vel = box_states[:, 7:10]
    box_ang_vel = box_states[:, 10:13]

    local_box_pos = box_pos - root_pos
    local_box_pos = quat_rotate(heading_rot, local_box_pos)

    box_standing_points_xy = box_standing_points[:, 0:3]
    box_standing_points_xy[:, 2] = 0.0
    local_box_standing_points_pos = box_standing_points_xy - root_pos
    local_box_standing_points_pos = quat_rotate(
        heading_rot, local_box_standing_points_pos)

    local_box_rot = quat_mul(heading_rot, box_rot)
    local_box_rot_obs = torch_utils.quat_to_tan_norm(local_box_rot)

    local_box_vel = quat_rotate(heading_rot, box_vel)
    local_box_ang_vel = quat_rotate(heading_rot, box_ang_vel)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos_obs = quat_rotate(heading_rot, local_tar_pos)
    local_tar_rot = quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    tar_standing_points_xy = tar_standing_points[:, 0:3]
    tar_standing_points_xy[:, 2] = 0.0
    local_tar_standing_points_pos = tar_standing_points_xy - root_pos
    local_tar_standing_points_pos = quat_rotate(
        heading_rot, local_tar_standing_points_pos)

    lfus = box_bps[0]
    lfds = box_bps[1]
    lbus = box_bps[2]
    lbds = box_bps[3]

    rfus = box_bps[4]
    rfds = box_bps[5]
    rbus = box_bps[6]
    rbds = box_bps[7]

    box_local_lfus_pos = convert_static_point_to_local_observation(
        lfus, root_states, box_pos, box_rot)
    box_local_lfds_pos = convert_static_point_to_local_observation(
        lfds, root_states, box_pos, box_rot)
    box_local_lbus_pos = convert_static_point_to_local_observation(
        lbus, root_states, box_pos, box_rot)
    box_local_lbds_pos = convert_static_point_to_local_observation(
        lbds, root_states, box_pos, box_rot)

    box_local_rfus_pos = convert_static_point_to_local_observation(
        rfus, root_states, box_pos, box_rot)
    box_local_rfds_pos = convert_static_point_to_local_observation(
        rfds, root_states, box_pos, box_rot)
    box_local_rbus_pos = convert_static_point_to_local_observation(
        rbus, root_states, box_pos, box_rot)
    box_local_rbds_pos = convert_static_point_to_local_observation(
        rbds, root_states, box_pos, box_rot)

    # add bps for tar

    tar_local_lfus_pos = convert_static_point_to_local_observation(
        lfus, root_states, tar_pos, tar_rot)
    tar_local_lfds_pos = convert_static_point_to_local_observation(
        lfds, root_states, tar_pos, tar_rot)
    tar_local_lbus_pos = convert_static_point_to_local_observation(
        lbus, root_states, tar_pos, tar_rot)
    tar_local_lbds_pos = convert_static_point_to_local_observation(
        lbds, root_states, tar_pos, tar_rot)

    tar_local_rfus_pos = convert_static_point_to_local_observation(
        rfus, root_states, tar_pos, tar_rot)
    tar_local_rfds_pos = convert_static_point_to_local_observation(
        rfds, root_states, tar_pos, tar_rot)
    tar_local_rbus_pos = convert_static_point_to_local_observation(
        rbus, root_states, tar_pos, tar_rot)
    tar_local_rbds_pos = convert_static_point_to_local_observation(
        rbds, root_states, tar_pos, tar_rot)

    obs = torch.cat([local_box_pos, local_box_rot_obs,
                    local_box_vel, local_box_ang_vel], dim=-1)
    obs = torch.cat([box_local_lfus_pos, box_local_lfds_pos, box_local_lbus_pos, box_local_lbds_pos,
                    box_local_rfus_pos, box_local_rfds_pos, box_local_rbus_pos, box_local_rbds_pos, obs], dim=-1)
    obs = torch.cat([local_box_standing_points_pos, obs], dim=-1)
    obs = torch.cat([local_tar_pos_obs, local_tar_rot_obs, obs], dim=-1)
    obs = torch.cat([tar_local_lfus_pos, tar_local_lfds_pos, tar_local_lbus_pos, tar_local_lbds_pos,
                    tar_local_rfus_pos, tar_local_rfds_pos, tar_local_rbus_pos, tar_local_rbds_pos, obs], dim=-1)
    # obs = torch.cat([torch.unsqueeze(density, -1), obs], dim=-1)
    # Do not use standing points for targets
    # obs = torch.cat(local_tar_standing_points_pos, obs)

    return obs


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids,
                           rigid_body_pos, box_contact_forces, lift_body_ids,
                           max_episode_length, enable_early_termination,
                           termination_heights,
                           box_pos, prev_box_pos, dt_tensor, hand_positions):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int, bool, Tensor,Tensor,Tensor,Tensor,Tensor) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 1.0
    box_vel_threshold = 1.0
    box_height_threshold = 0.3

    terminated = torch.zeros_like(reset_buf)

    # Early termination logic based on contact forces and body positions
    if enable_early_termination:
        # Mask the contact forces of the lifting body parts so they're not considered
        fall_masked_contact_buf = contact_buf.clone()
        fall_masked_contact_buf[:, contact_body_ids, :] = 0

        # Check if any body parts are making contact with a force above a minimal threshold
        # to determine if a fall contact has occurred.
        fall_contact = torch.any(
            torch.abs(fall_masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        # Check if the body height of any body parts is below a certain threshold
        # to determine if a fall due to height has occurred.
        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        # Do not consider lifting body parts for the height check
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        # Combine the conditions to determine if the humanoid has fallen
        has_fallen = torch.logical_and(fall_contact, fall_height)

        # check if the humanoid is kicking the box
        # box_has_contact_horizontal = torch.any(
        #     torch.abs(box_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)

        # check if the humanoid is kicking the box
        box_height = box_pos[..., 2]
        delta_box_pos = box_pos - prev_box_pos
        box_vel = delta_box_pos / dt_tensor
        box_vel_xy = box_vel[..., 0:2]
        box_vel_xy_norm = torch.norm(box_vel_xy, dim=-1)
        box_has_velocity_horizontal = box_vel_xy_norm > box_vel_threshold
        box_low = box_height < box_height_threshold
        mean_hand_positions = hand_positions[..., 0:3].mean(dim=1)
        hand_high = mean_hand_positions[..., 2] > 0.5

        box_kicked = torch.logical_and(box_has_velocity_horizontal, box_low)
        box_kicked_with_hands_high = torch.logical_and(box_kicked, hand_high)

        # has_failed = has_fallen
        # if forbid the agents to kick box,the agents may not know what to do.
        has_failed = torch.logical_or(has_fallen, box_kicked_with_hands_high)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(
            has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1,
                        torch.ones_like(reset_buf), terminated)

    return reset, terminated


@torch.jit.script
def compute_walk_reward(root_pos, root_rot, prev_root_pos, box_standing_pos, dt):
    # encourage the agent to walk towards box standing points

    near_threshold = 0.04
    target_speed = 1.0  # target speed in m/s
    pos_err_scale = 2.0
    vel_err_scale = 2.0

    # compute r_walk_pos
    box_standing_points_pos = box_standing_pos[..., 0:2]
    box_pos_diff = box_standing_points_pos - root_pos[..., 0:2]
    box_pos_err = torch.sum(box_pos_diff * box_pos_diff, dim=-1)
    box_pos_reward = torch.exp(-pos_err_scale * box_pos_err)

    # compute r_walk_vel

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    box_dir = torch.nn.functional.normalize(box_pos_diff, dim=-1)
    box_dir_speed = torch.sum(box_dir * root_vel[..., :2], dim=-1)
    box_vel_err = target_speed - box_dir_speed
    box_vel_err = torch.clamp_min(box_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (box_vel_err * box_vel_err))
    speed_mask = box_dir_speed <= 0
    vel_reward[speed_mask] = 0

    # compute r_walk_face

    heading_rot = torch_utils.calc_heading_quat(root_rot)

    facing_dir = torch.zeros_like(root_pos[..., 0:3])
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)

    facing_err = torch.sum(box_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    # compute r_walk

    near_mask = box_pos_err <= near_threshold
    box_pos_reward[near_mask] = 1.0
    vel_reward[near_mask] = 1.0
    facing_reward[near_mask] = 1.0

    return box_pos_reward, vel_reward, facing_reward


@torch.jit.script
def compute_contact_reward(hand_positions, box_held_points, root_pos, box_standing_pos, box_pos, tar_pos):
    box_near_threshold = 0.09
    carry_dist_threshold = 0.04
    box_height_threshold = 0.4
    held_pos_err_scale = 5.0
    mean_hand_positions = hand_positions[..., 0:3].mean(dim=1)
    hand2box_diff = mean_hand_positions - box_held_points[..., 0:3]
    hands2box_pos_err = torch.sum(hand2box_diff * hand2box_diff, dim=-1)
    hands2box_reward = torch.exp(-held_pos_err_scale * hands2box_pos_err)
    # compute masks when walking to box
    # box_standing_points_pos = box_standing_pos[..., 0:2]
    # box_pos_diff = box_standing_points_pos - root_pos[..., 0:2]
    # box_pos_err = torch.sum(box_pos_diff * box_pos_diff, dim=-1)
    # box_near_mask = box_pos_err <= box_near_threshold
    # hands2box_reward[~box_near_mask] = 0.0
    # compute masks when putdown
    box_height = box_held_points[..., 2]
    target_state_diff = tar_pos - box_pos  # xyz
    target_pos_err_xy = torch.sum(target_state_diff[..., 0:2] ** 2, dim=-1)
    near_mask = target_pos_err_xy <= carry_dist_threshold  # near_mask
    near_and_low_mask = torch.logical_and(
        near_mask, box_height < box_height_threshold)
    hands2box_reward[near_and_low_mask] = 1.0
    return hands2box_reward


@torch.jit.script
def compute_height_reward(held_point_height):
    target_height = 0.8
    height_err_scale = 10.0
    box_height_diff = target_height - held_point_height
    height_reward = torch.exp(
        -height_err_scale * box_height_diff * box_height_diff)
    return height_reward


@torch.jit.script
def compute_carry_reward(root_pos, root_rot, box_pos, box_rot, prev_box_pos, target_pos, target_rot, held_point_height, dt_tensor):
    target_speed = 1.0  # target speed in m/s
    carry_dist_threshold = 0.04
    height_threshold = 0.6
    tar_pos_err_far_scale = 0.5
    target_pos_err_near_scale = 10.0
    carry_vel_err_scale = 2.0

    x_axis = torch.zeros_like(root_pos[..., 0:3])
    x_axis[..., 0] = 1.0

    # masks
    box_height = box_pos[..., 2]
    height_mask = box_height < height_threshold

    # compute r_carry_pos
    target_state_diff = target_pos - box_pos  # xyz
    target_pos_err_xy = torch.sum(target_state_diff[..., 0:2] ** 2, dim=-1)
    near_mask = target_pos_err_xy <= carry_dist_threshold  # near_mask
    target_pos_err_xyz = torch.sum(target_state_diff[..., 0:3] ** 2, dim=-1)
    target_pos_reward_far = torch.exp(-tar_pos_err_far_scale *
                                      target_pos_err_xy)
    target_pos_reward_near = torch.exp(-target_pos_err_near_scale *
                                       target_pos_err_xyz)

    far_and_low_mask = torch.logical_and(~near_mask, height_mask)
    target_pos_reward_far[far_and_low_mask] = 0.0
    target_pos_reward_near[far_and_low_mask] = 0.0
    target_pos_reward_far[near_mask] = 1.0

    # compute_r_carry_face
    tar_dir = target_pos[..., 0:2] - box_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    tar_dir_reverse = box_pos[..., 0:2] - target_pos[..., 0:2]
    tar_dir_reverse = torch.nn.functional.normalize(tar_dir_reverse, dim=-1)
    root_heading_rot = torch_utils.calc_heading_quat(root_rot)
    root_facing_dir = quat_rotate(root_heading_rot, x_axis)
    # check whether the marker is behind the agent
    # if target is in front of the agent, then the agent should walk towards the target
    # if target is behind the agent, then the agent should walk backward to the target
    front_mask = torch.sum(tar_dir * root_facing_dir[..., 0:2], dim=-1) > 0
    behind_mask = torch.sum(
        tar_dir_reverse * root_facing_dir[..., 0:2], dim=-1) > 0
    facing_err = torch.sum(tar_dir * root_facing_dir[..., 0:2], dim=-1)
    facing_err[behind_mask] = torch.sum(
        tar_dir_reverse * root_facing_dir[..., 0:2], dim=-1)[behind_mask]
    facing_reward = torch.clamp_min(facing_err, 0.0)
    facing_reward[height_mask] = 0.0
    facing_reward[near_mask] = 1.0

    # compute r_carry_vel
    delta_box_pos = box_pos - prev_box_pos
    box_vel = delta_box_pos / dt_tensor
    box_tar_dir_speed = torch.sum(
        tar_dir * box_vel[..., 0:2], dim=-1)
    tar_vel_err = target_speed - box_tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    tar_vel_reward = torch.exp(-carry_vel_err_scale *
                               (tar_vel_err * tar_vel_err))
    tar_speed_mask = box_tar_dir_speed <= 0
    tar_vel_reward[tar_speed_mask] = 0
    tar_vel_reward[height_mask] = 0.0

    # compute r_carry_dir
    # calculate the facing direction of the box
    box_facing_dir = quat_rotate(box_rot, x_axis)
    tar_facing_dir = quat_rotate(target_rot, x_axis)
    dir_err = torch.sum(
        box_facing_dir[..., 0:2] * tar_facing_dir[..., 0:2], dim=-1)  # xy;higher value indicating better alignment
    dir_reward = torch.clamp_min(dir_err, 0.0)
    dir_reward[~near_mask] = 0.0

    # compute r_putdown
    held_points_height = held_point_height - target_pos[..., 2]
    put_down_height_reward = torch.exp(
        -5.0 * held_points_height * held_points_height)
    put_down_height_reward[~near_mask] = 0
    return target_pos_reward_far, tar_vel_reward, target_pos_reward_near, facing_reward, dir_reward, put_down_height_reward


@torch.jit.script
def compute_task_finish(box_pos, tar_pos, success_threshold):
    # type: (Tensor, Tensor, float) -> Tensor
    pos_diff = tar_pos - box_pos
    pos_err = torch.norm(pos_diff, p=2, dim=-1)
    dist_mask = pos_err <= success_threshold
    return dist_mask


@torch.jit.script
def compute_box_raise_height(box_half_size, box_height):
    height_diff = box_height - box_half_size
    return height_diff
