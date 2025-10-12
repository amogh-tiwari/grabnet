import numpy as np
import os
import torch
import mano
import trimesh
from grabnet.tools.utils import aa2rotmat
from grabnet.tools.train_tools import point2point_signed

from easydict import EasyDict as edict
from grabnet.models.models import CoarseNet, RefineNet

import sys
sys.path.append("../object_manipulation")
from object_manipulation.utils.io_utils import save_3d_object, load_mesh_batch, params_to_hand, load_mano_rhand
from object_manipulation.utils.rot_utils import transform_meshes, rotmat2aa

# ------------CHANGE BOX ---------------- #
cfg = {
    'data_dir': '/scratch/clear/atiwari/datasets/grabnet_extract/data',
    'params_fp': '/scratch/clear/atiwari/datasets/grabnet_extract/data/test/grabnet_test.npz',
    'frame_names_fp': '/scratch/clear/atiwari/datasets/grabnet_extract/data/test/frame_names.npz',
    'best_cnet': 'logs/V02_uni3d_embeds/snapshots/TR00_E009_cnet.pt',
    'best_rnet': 'logs/V02_uni3d_embeds/snapshots/TR00_E014_rnet.pt',
    'device': 'cuda:0',
    'dtype': torch.float32,
    'mano_path': './assets/MANO_RIGHT.pkl',
    'obj_meshes_root': '/scratch/clear/atiwari/datasets/grabnet_extract/tools/object_meshes/contact_meshes/',
    'out_dir': 'comparision_results/V02_uni3d_embeds',
    'n_samples': 178
}
cfg = edict(cfg)

np.random.seed(cfg.get("seed", 42))
torch.manual_seed(cfg.get("seed", 42))

# idxs = [30338, 22818, 48315, 22494, 2909, 8247, 56511, 45432, 3955, 57551, 52884, 48560, 8006, 56209, 56546, 49605, 4385, 31975, 28719, 43759, 19044, 
#         61965, 37063, 48798, 38821, 50909, 18275, 23072, 56817, 24731, 49591, 14872, 44038, 44456, 47089, 65019, 37639, 36669, 6288, 14200, 21379, 17554, 
#         21821, 37362, 21090, 41566, 60682, 59206, 58220, 45174, 22575, 14696, 58506, 49805, 15539, 33860, 20016, 55320, 44922, 56755, 35064, 35373, 26778,
#         55970, 48992, 14902, 60271, 43277, 45417, 56945, 45358, 4264, 65122, 4623, 44385, 48656, 38741, 11868, 251, 10646, 1034, 18476, 1400, 40907, 25796,
#         29115, 61570, 32498, 60737, 55871, 59943, 37246, 8180, 62419, 41162, 33558, 4815, 35957, 52475, 62893]

idxs = np.load("/scratch/clear/atiwari/datasets/grabnet_subset/data/test/frame_names_one_sample_per_sequence.npz")['selected_idxs']

n_samples = cfg['n_samples']
# ------------CHANGE BOX ---------------- #

params_fp = cfg['params_fp']
frame_names_fp = cfg['frame_names_fp']
params = np.load(params_fp)
frame_names = np.load(frame_names_fp)['frame_names']

if idxs is None:
    idxs = np.random.choice(params['trans_rhand'].shape[0], cfg['n_samples'], replace=False)
    n_samples = len(idxs)
else:
    assert n_samples == len(idxs), f"n_samples ({n_samples}) do not match match len(idxs) ({len(idxs)})"

bs = len(idxs)
with torch.no_grad():
    # rh_model = mano.load(model_path=cfg.mano_path, 
    #                 model_type='mano',
    #                 is_rhand= True,
    #                 num_pca_comps=45,
    #                 batch_size=bs,
    #                 flat_hand_mean=True).to(cfg.device)
    rh_model = load_mano_rhand(mano_path=cfg.mano_path, bs=n_samples, device=cfg.device)

coarse_net = CoarseNet().to(cfg.device)
coarse_net.load_state_dict(torch.load(cfg.best_cnet, map_location=cfg.device), strict=False)
refine_net = RefineNet().to(cfg.device)
refine_net.load_state_dict(torch.load(cfg.best_rnet, map_location=cfg.device), strict=False)
coarse_net.eval()
refine_net.eval()
refine_net.rhm_train = rh_model

bps_object = []
verts_object = []
in_fps = []

for idx in idxs:
    frame_full_path = os.path.join(cfg.data_dir, frame_names[idx])
    frame_info = np.load(frame_full_path)
    ds_path_addnl = frame_full_path.replace("grabnet_extract/data/", "grabnet_processing/uni3d_embeds/")
    addnl_data = np.load(ds_path_addnl)

    # bps_object.append(frame_info['bps_object'])
    bps_object.append(addnl_data['embed_object_uni3d_b_ensembled'])

    verts_object.append(frame_info['verts_object'])
    obj_name = frame_full_path.split("/")[-2].split("_")[0]
    in_fps.append(os.path.join(cfg['obj_meshes_root'], obj_name + ".ply"))

bps_object = torch.tensor(np.stack(bps_object), device=cfg.device, dtype=cfg.dtype)
verts_object = torch.tensor(np.stack(verts_object), device=cfg.device, dtype=cfg.dtype)
trans_rhand = torch.as_tensor(params['trans_rhand'][idxs]).to(cfg['device'])
global_orient_rhand_rotmat = torch.as_tensor(params['global_orient_rhand_rotmat'][idxs]).to(cfg['device'])
fpose_rhand_rotmat = torch.as_tensor(params['fpose_rhand_rotmat'][idxs]).to(cfg['device'])
root_orient_obj_rotmat = torch.as_tensor(params['root_orient_obj_rotmat'][idxs]).to(cfg['device'])
trans_obj = torch.as_tensor(params['trans_obj'][idxs]).to(cfg['device'])

input_params = {
    'bps_object': bps_object,
    # 'trans_rhand': trans_rhand,
    # 'global_orient_rhand_rotmat': global_orient_rhand_rotmat,
    # 'fpose_rhand_rotmat': fpose_rhand_rotmat,
}

out_params_cnet = coarse_net.sample_poses(**input_params)
out_hand_verts_cnet = rh_model(**out_params_cnet).vertices
out_params_cnet['verts_rhand'] = out_hand_verts_cnet

_, h2o, _ = point2point_signed(out_hand_verts_cnet, verts_object)
rnet_input = {}
rnet_input['trans_rhand_f'] = out_params_cnet['transl']
rnet_input['global_orient_rhand_rotmat_f'] = aa2rotmat(out_params_cnet['global_orient']).view(-1, 3, 3)
rnet_input['fpose_rhand_rotmat_f'] = aa2rotmat(out_params_cnet['hand_pose']).view(-1, 15, 3, 3)
rnet_input['verts_object'] = verts_object
rnet_input['h2o_dist']= h2o.abs()

out_params_rnet = refine_net(**rnet_input)
out_hand_verts_rnet = rh_model(**out_params_rnet).vertices
out_params_rnet['verts_rhand'] = out_hand_verts_rnet

fn_prefixes = [f"{idx:04d}" for idx in idxs]
inp_params = {
                'transl': trans_rhand, 
                'global_orient': rotmat2aa(global_orient_rhand_rotmat.view(n_samples, 1, -1, 9)).view(n_samples, -1), 
                'hand_pose': rotmat2aa(fpose_rhand_rotmat.view(n_samples, 1, -1, 9)).view(n_samples, -1)
            } # dict only for visualization; DON'T USE FPOSE INFO AS INPUT !!!
verts_gt, _ = params_to_hand(rh_model, inp_params, save=True, out_dir=os.path.join(cfg['out_dir'], 'infer'), fn_prefix=fn_prefixes, fn_suffix="_gt", color='green')
save_3d_object(verts=out_hand_verts_cnet.detach().cpu().numpy(), faces=rh_model.faces, out_dir=os.path.join(cfg.out_dir, 'infer'), fn_prefix=fn_prefixes, fn_suffix="_cnet", color='cyan')
save_3d_object(verts=out_hand_verts_rnet.detach().cpu().numpy(), faces=rh_model.faces, out_dir=os.path.join(cfg.out_dir, 'infer'), fn_prefix=fn_prefixes, fn_suffix="_rnet", color='navy')
meshes = load_mesh_batch(in_fps)
transformed_meshes, transformed_verts, transformed_faces = transform_meshes(meshes, root_orient_obj_rotmat, trans_obj)
save_3d_object(verts=transformed_verts, faces=transformed_faces, out_dir=os.path.join(cfg['out_dir'], 'infer'), fn_prefix=fn_prefixes, fn_suffix="_obj_mesh", color='yellow')

for loop_idx, file_idx in enumerate(idxs):
    cnet_dict = {k:v[loop_idx].detach().cpu().numpy() for k,v in out_params_cnet.items()}
    rnet_dict = {k:v[loop_idx].detach().cpu().numpy() for k,v in out_params_rnet.items()}
    np.savez_compressed(os.path.join(cfg['out_dir'], 'infer', f'{file_idx}_cnet_params.npz'), **cnet_dict)
    np.savez_compressed(os.path.join(cfg['out_dir'], 'infer', f'{file_idx}_rnet_params.npz'), **rnet_dict)

# for i in range(out_hand_verts_cnet.shape[0]):
#     hand_mesh_cnet = trimesh.Trimesh(vertices=out_hand_verts_cnet[i].detach().cpu().numpy(), faces=rh_model.faces)
#     _ = hand_mesh_cnet.export(os.path.join(cfg.out_dir, 'infer', f"{idxs[i]:04d}_cnet.obj"))
#     hand_mesh_rnet = trimesh.Trimesh(vertices=out_hand_verts_rnet[i].detach().cpu().numpy(), faces=rh_model.faces)
#     _ = hand_mesh_rnet.export(os.path.join(cfg.out_dir, 'infer', f"{idxs[i]:04d}_rnet.obj"))

print(f"Results saved to {cfg.out_dir}")
