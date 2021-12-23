# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
# Contact: Vassilis choutas, vassilis.choutas@tuebingen.mpg.de

import glob
import os
import os.path as osp

import argparse
import pickle
import torch
import smplx

from cmd_parser import parse_config
# from human_body_prior.tools.model_loader import load_vposer
from human_body_prior.tools.model_loader import load_model 
from human_body_prior.models.vposer_model import VPoser

from utils import JointMapper
import pyrender
import trimesh
import cv2
import numpy as np
import PIL.Image as pil_img
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_folder', type=str, required=True,
        help='The pkl folder that contains the files that will be read')
    parser.add_argument('--img_folder', type=str, required=True,
        help='The folder that contains the images that will be read')


    args, remaining = parser.parse_known_args()

    pkl_paths = args.pkl_folder
    img_paths = args.img_folder

    args = parse_config(remaining)
    dtype = torch.float32
    use_cuda = args.get('use_cuda', True)
    if use_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model_type = args.get('model_type', 'smplx')
    print('Model type:', model_type)
    print(args.get('model_folder'))
    model_params = dict(model_path=args.get('model_folder'),
        #  joint_mapper=joint_mapper,
        create_global_orient=True,
        create_body_pose=not args.get('use_vposer'),
        create_betas=True,
        create_left_hand_pose=True,
        create_right_hand_pose=True,
        create_expression=True,
        create_jaw_pose=True,
        create_leye_pose=True,
        create_reye_pose=True,
        create_transl=False,
        dtype=dtype,
        num_betas=300,
        **args
    )

    model = smplx.create(**model_params)
    model = model.to(device=device)

    batch_size = args.get('batch_size', 1)
    use_vposer = args.get('use_vposer', True)
    vposer, pose_embedding = [None, ] * 2
    vposer_ckpt = args.get('vposer_ckpt', '')
    if use_vposer:
        pose_embedding = torch.zeros([batch_size, 32],
            dtype=dtype, device=device,
            requires_grad=True
        )

        vposer_ckpt = osp.expandvars(vposer_ckpt)
        # vposer, _ = load_vposer(vposer_ckpt, vp_model='snapshot')
        vposer, _ = load_model(
            vposer_ckpt, model_code=VPoser,
            remove_words_in_model_weights='vp_model.', disable_grad=True
        )
        vposer = vposer.to(device=device)
        vposer.eval()
    
    mv = pyrender.Viewer(pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=(0.3, 0.3, 0.3)), use_raymond_lighting=True)
    light_nodes = mv._create_raymond_lights()
    
    for pkl_path in tqdm.tqdm(os.listdir(pkl_paths), desc='Scenes'):
        for i, pkl_filename in enumerate(glob.glob(os.path.join(pkl_paths, pkl_path, "*.pkl"))):
            with open(pkl_filename, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            img_filename = glob.glob(os.path.join(img_paths, f"{pkl_path}.jpg"))[0]
            img = cv2.imread(img_filename)
            if use_vposer:
                with torch.no_grad():
                    pose_embedding[:] = torch.tensor(
                        data['body_pose'], device=device, dtype=dtype)

            est_params = {}
            for key, val in data.items():
                if key == 'body_pose' and use_vposer:
                    # body_pose = vposer.decode(
                    #     pose_embedding, output_type='aa').view(1, -1)
                    body_pose = (
                        vposer.decode(pose_embedding).get( 'pose_body')
                    ).reshape(1, -1) if use_vposer else None
                    if model_type == 'smpl':
                        wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                                dtype=body_pose.dtype,
                                                device=body_pose.device)
                        body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                    est_params['body_pose'] = body_pose
                else:
                    est_params[key] = torch.tensor(val, dtype=dtype, device=device)

            model_output = model(**est_params)
            vertices = model_output.vertices.detach().cpu().numpy().squeeze()
            
            out_mesh = trimesh.Trimesh(vertices, model.faces, process=False)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=(1.0, 1.0, 0.9, 1.0))
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            out_mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(
                out_mesh,
                material=material)

            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                ambient_light=(0.3, 0.3, 0.3))
            scene.add(mesh, 'mesh')

            camera_transl = est_params['camera_translation'].detach().cpu().numpy().squeeze()
            camera_transl[0] *= -1.0
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_transl
            focal_length = 5000
            H, W, _ = img.shape
            camera_center = torch.tensor([W, H], dtype=dtype) * 0.5
            camera_center = camera_center.detach().cpu().numpy().squeeze()
            camera = pyrender.camera.IntrinsicsCamera(
                fx=focal_length, fy=focal_length,
                cx=camera_center[0], cy=camera_center[1])
            scene.add(camera, pose=camera_pose)

            # mv = pyrender.Viewer(scene, use_raymond_lighting=True)
            # light_nodes = mv._create_raymond_lights()
            for node in light_nodes:
                scene.add_node(node)
            r = pyrender.OffscreenRenderer(viewport_width=W,
                                        viewport_height=H,
                                        point_size=1.0)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0

            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img / 255.0
            output_img = (color[:, :, :-1] * valid_mask +
                        (1 - valid_mask) * input_img)

            img = pil_img.fromarray((output_img[:,:,::-1] * 255).astype(np.uint8))
            img.save(pkl_filename.replace('.pkl', f'_{i}.jpg'))

