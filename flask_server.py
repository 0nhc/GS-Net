import sys
import yaml
import os
import torch
import numpy as np
import open3d as o3d

from flask import Flask, request, jsonify
from models.graspnet import GraspNet, pred_decode
from dataset.graspnet_dataset import minkowski_collate_fn
from utils.collision_detector import ModelFreeCollisionDetector
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image, get_workspace_mask
from graspnetAPI.graspnet_eval import GraspGroup


class GSNetFlaskServer:
    def __init__(self):
        # Read Params
        self._read_params()

        # Flask related
        self.app = Flask(__name__)
        self.setup_routes()

        # GSNet relateds
        self._load_checkpoint()

    
    def setup_routes(self):
        @self.app.route('/get_gsnet_grasp', methods=['POST'])
        def get_gsnet_grasp():
            try:
                # Get data from request, data should be a list of points
                data = request.json

                # Run GSNet inference
                gsnet_input, pcd = self._as_gsnet_input(data)
                # Get the best grasping pose
                gg = self._gsnet_inference(gsnet_input)
                gg_return = self._get_best_grasping_pose(gg)
                if(self.cfg["visualize"]):
                    self._visualize(gg, pcd)

                return jsonify(gg_return)
            except Exception as e:
                # Clear CUDA memory
                if(self.device.type == 'cuda'):
                    torch.cuda.empty_cache()
                # Reload checkpoint
                self._load_checkpoint()
                return jsonify({'error': str(e)})
                
    

    def run(self):
        # self.app.run(debug=True)
        self.app.run(debug=True, host='0.0.0.0')


    def _get_best_grasping_pose(self, gg):
        # Select the best grasping pose
        gg = gg.nms()
        gg = gg.sort_by_score()[:self.cfg["max_num_grasps"]]
        gg_return = []
        for g in gg:
            p = g.translation
            r = g.rotation_matrix
            t = np.asarray([[r[0][0], r[0][1], r[0][2], p[0]],
                            [r[1][0], r[1][1], r[1][2], p[1]],
                            [r[2][0], r[2][1], r[2][2], p[2]],
                            [0,0,0,1]])
            score = g.score
            data = {'T': t.tolist(), 'score': score}
            gg_return.append(data)

        return gg_return


    def _as_gsnet_input(self, pcd):
        # sample points random
        cloud = np.asarray(pcd)
        if len(cloud) >= self.cfg["num_points"]:
            idxs = np.random.choice(len(cloud), self.cfg["num_points"], replace=False)
        else:
            idxs1 = np.arange(len(cloud))
            idxs2 = np.random.choice(len(cloud), self.cfg["num_points"] - len(cloud), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud = cloud[idxs]
        print(cloud.shape)
        ret_dict = {'point_clouds': cloud.astype(np.float32),
                    'coors': cloud.astype(np.float32) / self.cfg["voxel_size"],
                    'feats': np.ones_like(cloud).astype(np.float32),
                    }
        return ret_dict, cloud
    

    def _load_checkpoint(self):
        # Load checkpoint
        self.net = GraspNet(seed_feat_dim=self.cfg["seed_feat_dim"], is_training=False)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        checkpoint = torch.load(self.cfg["checkpoint_path"])
        self.net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)" % (self.cfg["checkpoint_path"], start_epoch))

        # Set to eval mode
        self.net.eval()
        

    def _gsnet_inference(self, data_input):
        # Load data
        batch_data = minkowski_collate_fn([data_input])

        # Move data to device
        for key in batch_data:
            if 'list' in key:
                for i in range(len(batch_data[key])):
                    for j in range(len(batch_data[key][i])):
                        batch_data[key][i][j] = batch_data[key][i][j].to(self.device)
            else:
                batch_data[key] = batch_data[key].to(self.device)
        # Forward pass
        with torch.no_grad():
            end_points = self.net(batch_data)
            grasp_preds = pred_decode(end_points)
        
        # Get predictions
        preds = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(preds)

        # collision detection
        if self.cfg["collision_thresh"] > 0:
            cloud = data_input['point_clouds']
            mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.cfg["voxel_size_cd"])
            collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.cfg["collision_thresh"])
            gg = gg[~collision_mask]

        return gg
    

    def _read_params(self):
        cfg_path = sys.path[0]+'/config/gsnet_flask_server.yaml'
        with open(cfg_path, 'r') as config:
            self.cfg = yaml.safe_load(config)

    def _visualize(self, grasp_group, pointcloud):
        grippers = grasp_group.to_open3d_geometry_list()
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(pointcloud.astype(np.float32))
        o3d.visualization.draw_geometries([cloud, *grippers])
    

gsnetflaskserver = GSNetFlaskServer()
gsnetflaskserver.run()
