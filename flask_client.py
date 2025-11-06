import requests
import numpy as np
import open3d as o3d

from PIL import Image


class GSNetFlaskClient:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self._base_url = base_url

        depth_path, color_path, intrinsics = self.DEMO_return_depth_data()
        pcd, colors = self.DEMO_get_pcd(depth_path, color_path, intrinsics)
        grasping_pose = self.request_grasping_pose(pcd)
        print(grasping_pose)
        
        # Automatically visualize if no error occurred
        if isinstance(grasping_pose, list) and len(grasping_pose) > 0:
            self.visualize_grasping_poses(pcd, grasping_pose, colors=colors)
        elif 'error' in grasping_pose:
            print(f"Error occurred: {grasping_pose['error']}")


    def request_grasping_pose(self, data):
        response = requests.post(f"{self._base_url}/get_gsnet_grasp", json=data)
        return response.json()
    

    def DEMO_return_depth_data(self):
        depth_path = "example_data/captured_depth_image.npy"
        color_path = "example_data/captured_color_image.png"
        intrinsics_path = "example_data/depth_intrinsics.npy"
        intrinsics = np.load(intrinsics_path)
        return depth_path, color_path, intrinsics


    def DEMO_get_pcd(self, depth_path, color_path, intrinsics):
        """
        Convert depth image to point cloud with RGB colors.
        
        Args:
            depth_path: Path to depth image (.npy file, already in meters)
            color_path: Path to RGB color image (.png file)
            intrinsics: Camera intrinsic matrix (3x3 numpy array)
            
        Returns:
            points: List of 3D points [[x, y, z], ...]
            colors: List of RGB colors [[r, g, b], ...] normalized to [0, 1]
        """
        # Load depth image (already in meters)
        depth = np.load(depth_path)
        
        # Ensure depth is 2D
        if len(depth.shape) > 2:
            depth = depth.squeeze()
        if len(depth.shape) != 2:
            raise ValueError(f"Depth image must be 2D, got shape {depth.shape}")
        
        # Get image dimensions from depth
        height, width = depth.shape
        
        # Load RGB color image
        color_img = np.asarray(Image.open(color_path), dtype=np.float32)
        # Convert to RGB if needed (handle RGBA or grayscale)
        if len(color_img.shape) == 2:
            # Grayscale, convert to RGB
            color_img = np.stack([color_img, color_img, color_img], axis=-1)
        elif color_img.shape[2] == 4:
            # RGBA, take only RGB
            color_img = color_img[:, :, :3]
        
        # Resize color image to match depth image dimensions if needed
        if color_img.shape[0] != height or color_img.shape[1] != width:
            color_pil = Image.fromarray((color_img * 255).astype(np.uint8) if color_img.max() <= 1.0 else color_img.astype(np.uint8))
            color_pil = color_pil.resize((width, height), Image.Resampling.LANCZOS)
            color_img = np.asarray(color_pil, dtype=np.float32)
            if color_img.max() > 1.0:
                color_img = color_img / 255.0
        
        # Normalize colors to [0, 1] if not already
        if color_img.max() > 1.0:
            color_img = color_img / 255.0
        
        # Extract camera intrinsics
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        # Create coordinate grids
        xmap, ymap = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert depth image to point cloud
        # Depth is already in meters, so no scaling needed
        points_z = depth.astype(np.float32)
        points_x = (xmap - cx) / fx * points_z
        points_y = (ymap - cy) / fy * points_z
        
        # Stack into point cloud
        points_3d = np.stack([points_x, points_y, points_z], axis=-1)
        
        # Filter out invalid points (where depth is 0, invalid, or greater than 1.5 meters)
        valid_mask = (points_z > 0) & (points_z <= 1.5) & np.isfinite(points_z) & np.isfinite(points_x) & np.isfinite(points_y)
        
        # Reshape to (H*W, 3) and filter
        points_3d = points_3d.reshape(-1, 3)
        colors_flat = color_img.reshape(-1, 3)
        valid_mask_flat = valid_mask.reshape(-1)
        
        # Keep only valid points
        points_valid = points_3d[valid_mask_flat]
        colors_valid = colors_flat[valid_mask_flat]
        
        # Convert to lists
        points = points_valid.tolist()
        colors = colors_valid.tolist()
        
        return points, colors
    
    def visualize_grasping_poses(self, point_cloud, grasping_poses, colors=None, fork_length=0.04, fork_width=0.06):
        """
        Visualize input point cloud and grasping poses as fork-like shapes.
        
        Args:
            point_cloud: List of points [[x, y, z], ...] or numpy array of shape (N, 3)
            grasping_poses: List of grasping poses, each containing:
                - 'T': 4x4 transformation matrix
                - 'score': grasp quality score
            colors: Optional list or numpy array of RGB colors for each point, shape (N, 3) with values in [0, 1]
            fork_length: Length of the fork prongs (default: 0.05m)
            fork_width: Width between the two prongs (default: 0.02m)
        """
        # Convert point cloud to numpy array if needed
        if isinstance(point_cloud, list):
            pcd_array = np.array(point_cloud)
        else:
            pcd_array = point_cloud
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_array.astype(np.float32))
        
        # Assign colors if available, otherwise use gray
        if colors is not None:
            # Ensure colors are in correct format (N, 3) with values in [0, 1]
            if isinstance(colors, list):
                colors_array = np.array(colors)
            else:
                colors_array = colors
            # Make sure colors are in [0, 1] range
            if colors_array.max() > 1.0:
                colors_array = colors_array / 255.0
            # Ensure shape matches
            if colors_array.shape[0] == pcd_array.shape[0]:
                pcd.colors = o3d.utility.Vector3dVector(colors_array.astype(np.float32))
            else:
                pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray if shape mismatch
        else:
            pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Gray color for point cloud
        
        geometries = [pcd]
        
        # Create fork-like shapes for each grasping pose
        for i, grasp in enumerate(grasping_poses):
            T = np.array(grasp['T'])
            score = grasp.get('score', 1.0)
            
            # Extract rotation and translation from transformation matrix
            R = T[:3, :3]
            t = T[:3, 3]
            
            # Create fork shape: ]--
            # Define fork points in local coordinate system
            # Fork shape: two parallel prongs (--) and a connecting part (])
            prong_length = fork_length
            prong_separation = fork_width
            handle_length = prong_length * 0.3
            stem_length = 0.05  # Length of the stem extending downward
            line_radius = 0.003  # Radius of the cylinder lines (3mm for thickness)
            
            # Points for the fork shape in local coordinates
            # Left prong (forward direction is +x)
            left_prong_start = np.array([0, -prong_separation/2, 0])
            left_prong_end = np.array([prong_length, -prong_separation/2, 0])
            
            # Right prong
            right_prong_start = np.array([0, prong_separation/2, 0])
            right_prong_end = np.array([prong_length, prong_separation/2, 0])
            
            # Connecting part (] shape)
            handle_start = np.array([-handle_length, -prong_separation/2, 0])
            handle_mid = np.array([-handle_length, 0, 0])
            handle_end = np.array([-handle_length, prong_separation/2, 0])
            
            # Stem at the bottom (extending downward in -z direction)
            stem_start = handle_mid.copy()
            stem_end = handle_mid + np.array([-stem_length, 0, 0])
            
            # Color based on score (red for high score, blue for low score)
            normalized_score = max(0, min(1, score))
            color = [normalized_score, 0, 1 - normalized_score]
            
            # Create cylinders for each line segment to make them thicker
            def create_cylinder_between_points(p1, p2, radius, color):
                """Create a cylinder mesh between two points."""
                # Calculate direction and length
                direction = p2 - p1
                length = np.linalg.norm(direction)
                if length < 1e-6:
                    return None
                
                # Create cylinder (default is along z-axis, centered at origin)
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
                
                # Rotate cylinder to align with direction
                # Default cylinder is along z-axis
                z_axis = np.array([0, 0, 1])
                direction_normalized = direction / length
                
                # Calculate rotation axis and angle
                if np.abs(np.dot(z_axis, direction_normalized)) > 0.99:
                    # Already aligned
                    rotation_matrix = np.eye(3)
                else:
                    # Calculate rotation using cross product
                    rotation_axis = np.cross(z_axis, direction_normalized)
                    rotation_axis = rotation_axis / (np.linalg.norm(rotation_axis) + 1e-8)
                    angle = np.arccos(np.clip(np.dot(z_axis, direction_normalized), -1, 1))
                    
                    # Create rotation matrix using Rodrigues' formula
                    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                                  [rotation_axis[2], 0, -rotation_axis[0]],
                                  [-rotation_axis[1], rotation_axis[0], 0]])
                    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
                
                # Apply rotation (cylinder is centered at origin)
                cylinder.rotate(rotation_matrix, center=[0, 0, 0])
                
                # Translate to midpoint (cylinder extends from -length/2 to +length/2 along z-axis)
                midpoint = (p1 + p2) / 2
                cylinder.translate(midpoint)
                
                # Color the cylinder
                cylinder.paint_uniform_color(color)
                
                return cylinder
            
            # Transform points to world coordinates
            left_prong_start_world = (R @ left_prong_start) + t
            left_prong_end_world = (R @ left_prong_end) + t
            right_prong_start_world = (R @ right_prong_start) + t
            right_prong_end_world = (R @ right_prong_end) + t
            handle_start_world = (R @ handle_start) + t
            handle_mid_world = (R @ handle_mid) + t
            handle_end_world = (R @ handle_end) + t
            stem_start_world = (R @ stem_start) + t
            stem_end_world = (R @ stem_end) + t
            
            # Create cylinders for each segment
            segments = [
                (left_prong_start_world, left_prong_end_world),
                (right_prong_start_world, right_prong_end_world),
                (handle_start_world, handle_mid_world),
                (handle_mid_world, handle_end_world),
                (left_prong_start_world, handle_start_world),
                (right_prong_start_world, handle_end_world),
                (stem_start_world, stem_end_world),  # Stem at the bottom
            ]
            
            for p1, p2 in segments:
                cylinder = create_cylinder_between_points(p1, p2, line_radius, color)
                if cylinder is not None:
                    geometries.append(cylinder)
        
        # Visualize
        o3d.visualization.draw_geometries(geometries)
                
        
    
client = GSNetFlaskClient()