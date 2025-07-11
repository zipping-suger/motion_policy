from typing import Union, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from geometry import TorchCuboids, TorchCylinders
from robofin.robots import FrankaRobot, FrankaRealRobot

from pathlib import Path
import logging

import torch
import numpy as np
import trimesh

from robofin.torch_urdf import TorchURDF
from robofin.robots import FrankaRobot


def transform_pointcloud(pc, transformation_matrix, in_place=True):
    """

    Parameters
    ----------
    pc: A pytorch tensor pointcloud, maybe with some addition dimensions.
        This should have shape N x [3 + M] where N is the number of points
        M could be some additional mask dimensions or whatever, but the
        3 are x-y-z
    transformation_matrix: A 4x4 homography

    Returns
    -------
    Mutates the pointcloud in place and transforms x, y, z according the homography

    """
    assert isinstance(pc, torch.Tensor)
    assert type(pc) == type(transformation_matrix)
    assert pc.ndim == transformation_matrix.ndim
    if pc.ndim == 3:
        N, M = 1, 2
    elif pc.ndim == 2:
        N, M = 0, 1
    else:
        raise Exception("Pointcloud must have dimension Nx3 or BxNx3")
    xyz = pc[..., :3]
    ones_dim = list(xyz.shape)
    ones_dim[-1] = 1
    ones_dim = tuple(ones_dim)
    homogeneous_xyz = torch.cat((xyz, torch.ones(ones_dim, device=xyz.device)), dim=M)
    transformed_xyz = torch.matmul(
        transformation_matrix, homogeneous_xyz.transpose(N, M)
    )
    if in_place:
        pc[..., :3] = transformed_xyz[..., :3, :].transpose(N, M)
        return pc
    return torch.cat((transformed_xyz[..., :3, :].transpose(N, M), pc[..., 3:]), dim=M)


class FrankaSampler:
    """
    This class allows for fast pointcloud sampling from the surface of a robot.
    At initialization, it loads a URDF and samples points from the mesh of each link.
    The points per link are based on the (very approximate) surface area of the link.

    Then, after instantiation, the sample method takes in a batch of configurations
    and produces pointclouds for each configuration by running FK on a subsample
    of the per-link pointclouds that are established at initialization.

    """

    def __init__(
        self,
        device,
        num_fixed_points=None,
        use_cache=False,
        default_prismatic_value=0.025,
        with_base_link=True,
    ):
        logging.getLogger("trimesh").setLevel("ERROR")
        self.num_fixed_points = num_fixed_points
        self.default_prismatic_value = default_prismatic_value
        self.with_base_link = with_base_link
        self._init_internal_(device, use_cache)

    def _init_internal_(self, device, use_cache):
        self.robot = TorchURDF.load(
            FrankaRobot.urdf, lazy_load_meshes=True, device=device
        )
        self.links = [l for l in self.robot.links if len(l.visuals)]
        if use_cache and self._init_from_cache_(device):
            return

        meshes = [
            trimesh.load(
                Path(FrankaRobot.urdf).parent / l.visuals[0].geometry.mesh.filename,
                force="mesh",
            )
            for l in self.links
        ]
        areas = [mesh.bounding_box_oriented.area for mesh in meshes]
        if self.num_fixed_points is not None:
            num_points = np.round(
                self.num_fixed_points * np.array(areas) / np.sum(areas)
            )
            num_points[0] += self.num_fixed_points - np.sum(num_points)
            assert np.sum(num_points) == self.num_fixed_points
        else:
            num_points = np.round(4096 * np.array(areas) / np.sum(areas))
        self.points = {}
        for ii in range(len(meshes)):
            pc = trimesh.sample.sample_surface(meshes[ii], int(num_points[ii]))[0]
            self.points[self.links[ii].name] = torch.as_tensor(
                pc, device=device
            ).unsqueeze(0)

        # If we made it all the way here with the use_cache flag set,
        # then we should be creating new cache files locally
        if use_cache:
            points_to_save = {
                k: tensor.squeeze(0).cpu().numpy() for k, tensor in self.points.items()
            }
            file_name = self._get_cache_file_name_()
            print(f"Saving new file to cache: {file_name}")
            np.save(file_name, points_to_save)

    def _get_cache_file_name_(self):
        if self.num_fixed_points is not None:
            return (
                FrankaRobot.pointcloud_cache
                / f"fixed_point_cloud_{self.num_fixed_points}.npy"
            )
        else:
            return FrankaRobot.pointcloud_cache / "full_point_cloud.npy"

    def _init_from_cache_(self, device):
        file_name = self._get_cache_file_name_()
        if not file_name.is_file():
            return False

        points = np.load(
            file_name,
            allow_pickle=True,
        )
        self.points = {
            key: torch.as_tensor(pc, device=device).unsqueeze(0)
            for key, pc in points.item().items()
        }
        return True

    def end_effector_pose(self, config, frame="right_gripper"):
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                self.default_prismatic_value
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.link_fk_batch(cfg, use_names=True)
        return fk[frame]

    def sample_end_effector(self, poses, num_points, frame="right_gripper"):
        """
        An internal method--separated so that the public facing method can
        choose whether or not to have gradients
        """
        assert poses.ndim in [2, 3]
        assert frame == "right_gripper", "Other frames not yet suppported"
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)
        default_cfg = torch.zeros((1, 9), device=poses.device)
        default_cfg[0, 7:] = self.default_prismatic_value
        fk = self.robot.visual_geometry_fk_batch(default_cfg)
        eff_link_names = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]

        # This logic could break--really need a way to make sure that the
        # ordering is correct
        values = [
            list(fk.values())[idx]
            for idx, l in enumerate(self.links)
            if l.name in eff_link_names
        ]
        end_effector_links = [l for l in self.links if l.name in eff_link_names]
        assert len(end_effector_links) == len(values)
        fk_transforms = {}
        fk_points = []
        gripper_T_hand = torch.as_tensor(
            FrankaRobot.EFF_T_LIST[("panda_hand", "right_gripper")].inverse.matrix
        ).type_as(poses)
        # Could just invert the matrix, but matrix inversion is not implemented for half-types
        inverse_hand_transform = torch.zeros_like(values[0])
        inverse_hand_transform[:, -1, -1] = 1
        inverse_hand_transform[:, :3, :3] = values[0][:, :3, :3].transpose(1, 2)
        inverse_hand_transform[:, :3, -1] = -torch.matmul(
            inverse_hand_transform[:, :3, :3], values[0][:, :3, -1].unsqueeze(-1)
        ).squeeze(-1)
        right_gripper_transform = gripper_T_hand.unsqueeze(0) @ inverse_hand_transform
        for idx, l in enumerate(end_effector_links):
            fk_transforms[l.name] = values[idx]
            pc = transform_pointcloud(
                self.points[l.name].type_as(poses),
                (right_gripper_transform @ fk_transforms[l.name]),
                in_place=True,
            )
            fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        pc = transform_pointcloud(pc.repeat(poses.size(0), 1, 1), poses)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]

    def sample(self, config, num_points=None):
        """
        Samples points from the surface of the robot by calling fk.

        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of
            actuated joints.
            For example, if using the Franka, M is 9
        num_points : Number of points desired

        Returns
        -------
        N x num points x 3 pointcloud of robot points

        """
        assert bool(self.num_fixed_points is None) ^ bool(num_points is None)
        if config.ndim == 1:
            config = config.unsqueeze(0)
        cfg = torch.cat(
            (
                config,
                self.default_prismatic_value
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        fk_transforms = {}
        fk_points = []
        for idx, l in enumerate(self.links):
            if l.name == "panda_link0" and not self.with_base_link:
                continue
            fk_transforms[l.name] = values[idx]
            pc = transform_pointcloud(
                self.points[l.name]
                .float()
                .repeat((fk_transforms[l.name].shape[0], 1, 1)),
                fk_transforms[l.name],
                in_place=True,
            )
            fk_points.append(pc)
        pc = torch.cat(fk_points, dim=1)
        if num_points is None:
            return pc
        return pc[:, np.random.choice(pc.shape[1], num_points, replace=False), :]
    
    def sample_per_link(self, config, total_points=None):
        """
        Samples points from each link's surface separately, distributing points proportionally
        to each link's surface area, and returns them in a dictionary.
        
        Parameters
        ----------
        config : Tensor of length (M,) or (N, M) where M is the number of
            actuated joints. For example, if using the Franka, M is 9
        total_points : Total number of points to sample across all links (optional)
            If None, uses all pre-sampled points for each link
        
        Returns
        -------
        Dictionary where keys are link names and values are point clouds:
        - If input is (M,): returns dict of (K, 3) tensors where K is num points for that link
        - If input is (N, M): returns dict of (N, K, 3) tensors where K is num points for that link
        """
        if config.ndim == 1:
            config = config.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
            
        cfg = torch.cat(
            (
                config,
                self.default_prismatic_value
                * torch.ones((config.shape[0], 2), device=config.device),
            ),
            dim=1,
        )
        fk = self.robot.visual_geometry_fk_batch(cfg)
        values = list(fk.values())
        assert len(self.links) == len(values)
        
        # Get relative sizes of each link (based on initialization)
        link_sizes = {l.name: self.points[l.name].shape[1] for l in self.links 
                    if not (l.name == "panda_link0" and not self.with_base_link)}
        total_fixed_points = sum(link_sizes.values())
        
        per_link_pcs = {}
        for idx, l in enumerate(self.links):
            if l.name == "panda_link0" and not self.with_base_link:
                continue
                
            # Get all pre-sampled points for this link
            link_pc = self.points[l.name].float().repeat((values[idx].shape[0], 1, 1))
            
            # Transform points using FK
            transformed_pc = transform_pointcloud(
                link_pc,
                values[idx],
                in_place=True,
            )
            
            # Subsample proportionally if total_points is specified
            if total_points is not None:
                # Calculate how many points this link should get
                link_points = max(1, int(round(
                    total_points * (link_sizes[l.name] / total_fixed_points)
                )))
                
                if transformed_pc.shape[1] < link_points:
                    # If we don't have enough pre-sampled points, use all we have
                    link_points = transformed_pc.shape[1]
                
                if link_points < transformed_pc.shape[1]:
                    transformed_pc = transformed_pc[
                        :, 
                        np.random.choice(transformed_pc.shape[1], link_points, replace=False), 
                        :
                    ]
            
            # Squeeze if input was 1D
            if squeeze_output:
                transformed_pc = transformed_pc.squeeze(0)
                
            per_link_pcs[l.name] = transformed_pc
        
        return per_link_pcs
    
    
def convert_robotB_to_robotA(pose_B):
    """
    Robot B is the Franka robot defined in robofin, which has an additional tool transformation
    Robot A is the Franka robot defined in Issac Lab
    
    Converts Robot B's pose (4x4 matrix) to match Robot A's convention
    by removing the additional tool transformation.
    
    Args:
        pose_B: 4x4 transformation matrix from Robot B
    
    Returns:
        T_A: 4x4 transformation matrix in Robot A's convention
    """
    # Extract rotation matrix and translation vector
    R_B = pose_B[:3, :3]
    p_B = pose_B[:3, 3]
    
    # Compute corrected rotation: 
    # Negate first two columns (180° z-rotation compensation)
    R_A = R_B.copy()
    R_A[:, 0] = -R_A[:, 0]  # Negate x-axis
    R_A[:, 1] = -R_A[:, 1]  # Negate y-axis
    
    # Compute corrected translation:
    # Remove tool offset along z-axis
    w = R_B[:, 2]  # Z-axis direction vector
    p_A = p_B - 0.1 * w
    
    # Build corrected transformation matrix
    T_A = np.eye(4)
    T_A[:3, :3] = R_A
    T_A[:3, 3] = p_A
    
    return T_A


def _normalize_franka_joints_numpy(
    batch_trajectory: np.ndarray,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> np.ndarray:
    """
    Normalizes joint angles to be within a specified range according to the Franka's
    joint limits. This is the numpy version

    :param batch_trajectory np.ndarray: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The new limits to map to
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype np.ndarray: An array with the same dimensions as the input
    """
    robot = FrankaRealRobot if use_real_constraints else FrankaRobot
    franka_limits = robot.JOINT_LIMITS
    assert (
        (batch_trajectory.ndim == 1 and batch_trajectory.shape[0] == robot.DOF)
        or (batch_trajectory.ndim == 2 and batch_trajectory.shape[1] == robot.DOF)
        or (batch_trajectory.ndim == 3 and batch_trajectory.shape[2] == robot.DOF)
    )
    normalized = (batch_trajectory - franka_limits[:, 0]) / (
        franka_limits[:, 1] - franka_limits[:, 0]
    ) * (limits[1] - limits[0]) + limits[0]
    return normalized


def _normalize_franka_joints_torch(
    batch_trajectory: torch.Tensor,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> torch.Tensor:
    """
    Normalizes joint angles to be within a specified range according to the Franka's
    joint limits. This is the torch version

    :param batch_trajectory torch.Tensor: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The new limits to map to
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype torch.Tensor: A tensor with the same dimensions as the input
    """
    assert isinstance(batch_trajectory, torch.Tensor)
    robot = FrankaRealRobot if use_real_constraints else FrankaRobot
    franka_limits = torch.as_tensor(robot.JOINT_LIMITS).type_as(batch_trajectory)
    assert (
        (batch_trajectory.ndim == 1 and batch_trajectory.size(0) == robot.DOF)
        or (batch_trajectory.ndim == 2 and batch_trajectory.size(1) == robot.DOF)
        or (batch_trajectory.ndim == 3 and batch_trajectory.size(2) == robot.DOF)
    )
    return (batch_trajectory - franka_limits[:, 0]) / (
        franka_limits[:, 1] - franka_limits[:, 0]
    ) * (limits[1] - limits[0]) + limits[0]


def normalize_franka_joints(
    batch_trajectory: Union[np.ndarray, torch.Tensor],
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Normalizes joint angles to be within a specified range according to the Franka's
    joint limits. This is semantic sugar that dispatches to the correct implementation.

    :param batch_trajectory Union[np.ndarray, torch.Tensor]: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The new limits to map to
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype Union[np.ndarray, torch.Tensor]: A tensor or numpy array with the same dimensions
                                            and type as the input
    :raises NotImplementedError: Raises an error if another data type (e.g. a list) is passed in
    """
    if isinstance(batch_trajectory, torch.Tensor):
        return _normalize_franka_joints_torch(
            batch_trajectory, limits=limits, use_real_constraints=True
        )
    elif isinstance(batch_trajectory, np.ndarray):
        return _normalize_franka_joints_numpy(
            batch_trajectory, limits=limits, use_real_constraints=True
        )
    else:
        raise NotImplementedError("Only torch.Tensor and np.ndarray implemented")


def _unnormalize_franka_joints_numpy(
    batch_trajectory: np.ndarray,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> np.ndarray:
    """
    Unnormalizes joint angles from a specified range back into the Franka's joint limits.
    This is the numpy version and the inverse of `_normalize_franka_joints_numpy`.

    :param batch_trajectory np.ndarray: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The current limits to map to the joint limits
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype np.ndarray: An array with the same dimensions as the input
    """
    robot = FrankaRealRobot if use_real_constraints else FrankaRobot
    franka_limits = robot.JOINT_LIMITS
    assert (
        (batch_trajectory.ndim == 1 and batch_trajectory.shape[0] == robot.DOF)
        or (batch_trajectory.ndim == 2 and batch_trajectory.shape[1] == robot.DOF)
        or (batch_trajectory.ndim == 3 and batch_trajectory.shape[2] == robot.DOF)
    )
    assert np.all(batch_trajectory >= limits[0])
    assert np.all(batch_trajectory <= limits[1])
    franka_limit_range = franka_limits[:, 1] - franka_limits[:, 0]
    franka_lower_limit = franka_limits[:, 0]
    for _ in range(batch_trajectory.ndim - 1):
        franka_limit_range = franka_limit_range[np.newaxis, ...]
        franka_lower_limit = franka_lower_limit[np.newaxis, ...]
    unnormalized = (batch_trajectory - limits[0]) * franka_limit_range / (
        limits[1] - limits[0]
    ) + franka_lower_limit

    return unnormalized


def _unnormalize_franka_joints_torch(
    batch_trajectory: torch.Tensor,
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> torch.Tensor:
    """
    Unnormalizes joint angles from a specified range back into the Franka's joint limits.
    This is the torch version and the inverse of `_normalize_franka_joints_torch`.

    :param batch_trajectory torch.Tensor: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The current limits to map to the joint limits
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype torch.Tensor: A tensor with the same dimensions as the input
    """
    assert isinstance(batch_trajectory, torch.Tensor)
    robot = FrankaRealRobot if use_real_constraints else FrankaRobot
    franka_limits = torch.as_tensor(robot.JOINT_LIMITS).type_as(batch_trajectory)
    dof = franka_limits.size(0)
    assert (
        (batch_trajectory.ndim == 1 and batch_trajectory.size(0) == dof)
        or (batch_trajectory.ndim == 2 and batch_trajectory.size(1) == dof)
        or (batch_trajectory.ndim == 3 and batch_trajectory.size(2) == dof)
    )
    assert torch.all(batch_trajectory >= limits[0])
    assert torch.all(batch_trajectory <= limits[1])
    franka_limit_range = franka_limits[:, 1] - franka_limits[:, 0]
    franka_lower_limit = franka_limits[:, 0]
    for _ in range(batch_trajectory.ndim - 1):
        franka_limit_range = franka_limit_range.unsqueeze(0)
        franka_lower_limit = franka_lower_limit.unsqueeze(0)
    return (batch_trajectory - limits[0]) * franka_limit_range / (
        limits[1] - limits[0]
    ) + franka_lower_limit


def unnormalize_franka_joints(
    batch_trajectory: Union[np.ndarray, torch.Tensor],
    limits: Tuple[float, float] = (-1, 1),
    use_real_constraints: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Unnormalizes joint angles from a specified range back into the Franka's joint limits.
    This is semantic sugar that dispatches to the correct implementation, the inverse of
    `normalize_franka_joints`.

    :param batch_trajectory Union[np.ndarray, torch.Tensor]: A batch of trajectories. Can have dims
                                        [7] if a single configuration
                                        [B, 7] if a batch of configurations
                                        [B, T, 7] if a batched time-series of configurations
    :param limits Tuple[float, float]: The current limits to map to the joint limits
    :param use_real_constraints bool: If true, use the empirically determined joint limits
                                      (this is unpublished--just found by monkeying around
                                      with the robot).
                                      If false, use the published joint limits from Franka
    :rtype Union[np.ndarray, torch.Tensor]: A tensor or numpy array with the same dimensions
                                            and type as the input
    :raises NotImplementedError: Raises an error if another data type (e.g. a list) is passed in
    """
    if isinstance(batch_trajectory, torch.Tensor):
        return _unnormalize_franka_joints_torch(
            batch_trajectory, limits=limits, use_real_constraints=use_real_constraints
        )
    elif isinstance(batch_trajectory, np.ndarray):
        return _unnormalize_franka_joints_numpy(
            batch_trajectory, limits=limits, use_real_constraints=use_real_constraints
        )
    else:
        raise NotImplementedError("Only torch.Tensor and np.ndarray implemented")


def collision_loss(
    input_pc: torch.Tensor,
    cuboid_centers: torch.Tensor,
    cuboid_dims: torch.Tensor,
    cuboid_quaternions: torch.Tensor,
    cylinder_centers: torch.Tensor,
    cylinder_radii: torch.Tensor,
    cylinder_heights: torch.Tensor,
    cylinder_quaternions: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the hinge loss, calculating whether the robot (represented as a
    point cloud) is in collision with any obstacles in the scene. Collision
    here actually means within 3cm of the obstacle--this is to provide stronger
    gradient signal to encourage the robot to move out of the way. Also, some of the
    primitives can have zero volume (i.e. a dim is zero for cuboids or radius or height is zero for cylinders).
    If these are zero volume, they will have infinite sdf values (and therefore be ignored by the loss).

    :param input_pc torch.Tensor: Points sampled from the robot's surface after it
                                  is placed at the network's output prediction. Has dim [B, N, 3]
    :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
    :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
    :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
    :rtype torch.Tensor: Returns the loss value aggregated over the batch
    """

    cuboids = TorchCuboids(
        cuboid_centers,
        cuboid_dims,
        cuboid_quaternions,
    )
    cylinders = TorchCylinders(
        cylinder_centers,
        cylinder_radii,
        cylinder_heights,
        cylinder_quaternions,
    )
    sdf_values = torch.minimum(cuboids.sdf(input_pc), cylinders.sdf(input_pc))
    return F.hinge_embedding_loss(
        sdf_values,
        -torch.ones_like(sdf_values),
        margin=0.03,
        reduction="mean",
    )


def minimal_collision_distance(
    input_pc: torch.Tensor,
    cuboid_centers: torch.Tensor,
    cuboid_dims: torch.Tensor,
    cuboid_quaternions: torch.Tensor,
    cylinder_centers: torch.Tensor,
    cylinder_radii: torch.Tensor,
    cylinder_heights: torch.Tensor,
    cylinder_quaternions: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the minimal signed distance from the input point cloud to any obstacle
    (cuboid or cylinder) in the scene for each batch element.

    :param input_pc torch.Tensor: Points sampled from the robot's surface after it
                                  is placed at the network's output prediction. Has dim [B, N, 3]
    :param cuboid_centers torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_dims torch.Tensor: Has dim [B, M1, 3]
    :param cuboid_quaternions torch.Tensor: Has dim [B, M1, 4]. Quaternion is formatted as w, x, y, z.
    :param cylinder_centers torch.Tensor: Has dim [B, M2, 3]
    :param cylinder_radii torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_heights torch.Tensor: Has dim [B, M2, 1]
    :param cylinder_quaternions torch.Tensor: Has dim [B, M2, 4]. Quaternion is formatted as w, x, y, z.
    :rtype torch.Tensor: Returns the minimal distance for each batch element, shape [B]
    """
    cuboids = TorchCuboids(
        cuboid_centers,
        cuboid_dims,
        cuboid_quaternions,
    )
    cylinders = TorchCylinders(
        cylinder_centers,
        cylinder_radii,
        cylinder_heights,
        cylinder_quaternions,
    )
    # Compute SDFs: [B, N, M1] and [B, N, M2]
    cuboid_sdf = cuboids.sdf(input_pc)      # [B, N, M1]
    cylinder_sdf = cylinders.sdf(input_pc)  # [B, N, M2]

    # Minimal SDF for each point to any cuboid/cylinder: [B, N]
    min_cuboid_sdf = cuboid_sdf.min(dim=-1).values
    min_cylinder_sdf = cylinder_sdf.min(dim=-1).values

    # Minimal SDF for each point to any obstacle: [B, N]
    min_sdf = torch.minimum(min_cuboid_sdf, min_cylinder_sdf)

    # Minimal SDF across all points in the point cloud: [B]
    min_dist = min_sdf.min(dim=-1).values

    return min_dist


def links_obs_dist(
    link_pointclouds: dict,  # Dictionary of {link_name: torch.Tensor(B, N, 3)}
    cuboid_centers: torch.Tensor,  # shape (B, M1, 3)
    cuboid_dims: torch.Tensor,  # shape (B, M1, 3)
    cuboid_quaternions: torch.Tensor,  # shape (B, M1, 4) [w, x, y, z]
    cylinder_centers: torch.Tensor,  # shape (B, M2, 3)
    cylinder_radii: torch.Tensor,  # shape (B, M2, 1)
    cylinder_heights: torch.Tensor,  # shape (B, M2, 1)
    cylinder_quaternions: torch.Tensor,  # shape (B, M2, 4) [w, x, y, z]
    debug: bool = False
):
    """
    Computes minimal distances between robot links and obstacles using PyTorch for efficient batch processing.
    
    Args:
        debug: If True, returns detailed debug information. If False, returns a tensor of shape (B, L)
               where L is the number of links.
    
    Returns:
        If debug=True:
        dict: {
            'min_distance': torch.Tensor,  # Minimal distance across all links (B,)
            'per_link': {  # Distances per link
                'link_name': {
                    'min_distance': torch.Tensor,  # (B,)
                    'closest_obstacle_type': str,
                    'closest_obstacle_idx': torch.Tensor  # (B,)
                },
                ...
            }
        }
        If debug=False:
        torch.Tensor: Minimal distances for each link, shape (B, L) where L is number of links
    """
    device = cuboid_centers.device
    batch_size = cuboid_centers.shape[0]
    # Remove unnecessary links
    links_to_remove = ['panda_link0', 'panda_link7', 'panda_leftfinger', 'panda_rightfinger']
    for k in links_to_remove:
        if k in link_pointclouds:
            del link_pointclouds[k]
    link_names = list(link_pointclouds.keys())
    num_links = len(link_names)
    
    # Initialize results
    if debug:
        results = {
            'min_distance': torch.full((batch_size,), float('inf'), device=device),
            'per_link': {}
        }
    else:
        link_dists = torch.full((batch_size, num_links), float('inf'), device=device)
    
    # Create obstacle primitives
    cuboids = TorchCuboids(cuboid_centers, cuboid_dims, cuboid_quaternions)
    cylinders = TorchCylinders(cylinder_centers, cylinder_radii, cylinder_heights, cylinder_quaternions)
    
    for link_idx, (link_name, pc) in enumerate(link_pointclouds.items()):
        # Compute distances to cuboids
        cuboid_dists = cuboids.sdf(pc)  # (B, N, M1)
        cuboid_min_dists = cuboid_dists.min(dim=-1).values.min(dim=-1).values  # (B,)
        
        # Compute distances to cylinders
        cylinder_dists = cylinders.sdf(pc)  # (B, N, M2)
        cylinder_min_dists = cylinder_dists.min(dim=-1).values.min(dim=-1).values  # (B,)
        
        # Get minimum between cuboid and cylinder distances
        link_min_dist = torch.minimum(cuboid_min_dists, cylinder_min_dists)
        
        if debug:
            # Determine closest obstacle type
            mask = cuboid_min_dists < cylinder_min_dists
            closest_type = 'cuboid' if mask.any() else 'cylinder'
            
            # Store detailed results
            results['per_link'][link_name] = {
                'min_distance': link_min_dist,
                'closest_obstacle_type': closest_type,
                # Note: Original implementation didn't track closest obstacle idx
            }
            
            # Update global minimum
            results['min_distance'] = torch.minimum(results['min_distance'], link_min_dist)
        else:
            link_dists[:, link_idx] = link_min_dist
    
    return results if debug else link_dists


def compute_pose_loss_quat(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes position and rotation loss between predicted and target end-effector poses.

    Args:
        pred_pose (torch.Tensor): Predicted pose (B,4,4)
        target_pose (torch.Tensor): Target pose (B,7): [x, y, z, qw, qx, qy, qz]

    Returns:
        position_loss (torch.Tensor): (B,) squared Euclidean position loss
        rotation_loss (torch.Tensor): (B,) geodesic rotation loss in radians
    """
    # Extract target position and quaternion
    target_pos = target_pose[:, 0:3]  # (B,3)
    target_quat = target_pose[:, 3:]  # (B,4) format: [qw, qx, qy, qz]

    # Normalize quaternion and convert to rotation matrix
    q_norm = torch.norm(target_quat, dim=1, keepdim=True)
    q_norm = torch.clamp(q_norm, min=1e-12)
    target_quat = target_quat / q_norm
    qw, qx, qy, qz = target_quat.unbind(dim=1)  # [qw, qx, qy, qz]

    # Compute rotation matrix from quaternion
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    xw, yw, zw = qx * qw, qy * qw, qz * qw

    R_target = torch.stack((
        1 - 2 * (yy + zz), 2 * (xy - zw),     2 * (xz + yw),
        2 * (xy + zw),     1 - 2 * (xx + zz), 2 * (yz - xw),
        2 * (xz - yw),     2 * (yz + xw),     1 - 2 * (xx + yy)
    ), dim=1).view(-1, 3, 3)  # (B,3,3)

    # Extract predicted rotation and translation
    R_pred = pred_pose[:, :3, :3]  # (B,3,3)
    t_pred = pred_pose[:, :3, 3]   # (B,3)

    # Positional loss (squared Euclidean)
    position_loss = torch.sum((t_pred - target_pos) ** 2, dim=1)  # (B,)

    # Rotational loss using geodesic distance (in radians)
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_target)  # (B,3,3)
    trace = torch.einsum('bii->b', R_diff)  # (B,)
    trace_clamped = torch.clamp((trace - 1) / 2, min=-1.0, max=1.0)
    rotation_loss = torch.acos(trace_clamped)  # (B,)

    return position_loss, rotation_loss


def compute_pose_loss_rotmat(
    pred_pose: torch.Tensor,
    target_pose: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes position and rotation loss between predicted and target end-effector poses.

    Args:
        pred_pose (torch.Tensor): Predicted pose (B, 4, 4)
        target_pose (torch.Tensor): Target pose (B, 12): [x, y, z, flattened 3x3 rotation matrix (row-major)]

    Returns:
        position_loss (torch.Tensor): (B,) squared Euclidean position loss
        rotation_loss (torch.Tensor): (B,) squared Frobenius norm between rotation matrices
    """
    # Extract target position and rotation matrix
    target_pos = target_pose[:, 0:3]  # (B, 3)
    target_rot = target_pose[:, 3:12].view(-1, 3, 3)  # (B, 3, 3)

    # Extract predicted rotation and translation
    pred_rot = pred_pose[:, :3, :3]  # (B, 3, 3)
    pred_pos = pred_pose[:, :3, 3]   # (B, 3)

    # Position loss (squared Euclidean distance)
    position_loss = torch.sum((pred_pos - target_pos) ** 2, dim=1)  # (B,)

    # Rotation loss (Chordal Distance = squared Frobenius norm)
    rotation_loss = torch.sum((pred_rot - target_rot) ** 2, dim=(1, 2))  # (B,)

    return position_loss, rotation_loss

