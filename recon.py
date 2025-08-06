import numpy as np
import os
import torch
import mano
import trimesh
from grabnet.tools.utils import aa2rotmat
from grabnet.tools.train_tools import point2point_signed

from easydict import EasyDict as edict
from grabnet.models.models import CoarseNet, RefineNet


# ------------CHANGE BOX ---------------- #
cfg = {
    'data_dir': '/scratch/clear/atiwari/datasets/grabnet_extract/data',
    'params_fp': '/scratch/clear/atiwari/datasets/grabnet_extract/data/test/grabnet_test.npz',
    'frame_names_fp': '/scratch/clear/atiwari/datasets/grabnet_extract/data/test/frame_names.npz',
    'best_cnet': 'ckpts/coarsenet.pt',
    'best_rnet': 'ckpts/refinenet.pt',
    'device': 'cuda:0',
    'dtype': torch.float32,
    'mano_path': './assets/MANO_RIGHT.pkl',
    'out_dir': 'outputs',
    'n_samples': 100
}
cfg = edict(cfg)

np.random.seed(cfg.get("seed", 42))
torch.manual_seed(cfg.get("seed", 42))

# idxs = [2909, 22494, 22818, 30338, 48315]
idxs = None
n_samples = cfg['n_samples']
# ------------CHANGE BOX ---------------- #

params_fp = cfg['params_fp']
frame_names_fp = cfg['frame_names_fp']
params = np.load(params_fp)
frame_names = np.load(frame_names_fp)['frame_names']

if idxs == None:
    idxs = np.random.choice(params['trans_rhand'].shape[0], cfg['n_samples'], replace=False)
    n_samples = len(idxs)
else:
    assert n_samples == len(idxs), f"n_samples ({n_samples}) do not match match len(idxs) ({len(idxs)})"

bs = len(idxs)
with torch.no_grad():
    rh_model = mano.load(model_path=cfg.mano_path, 
                    model_type='mano',
                    is_rhand= True,
                    num_pca_comps=45,
                    batch_size=bs,
                    flat_hand_mean=True).to(cfg.device)

coarse_net = CoarseNet().to(cfg.device)
coarse_net.load_state_dict(torch.load(cfg.best_cnet, map_location=cfg.device), strict=False)
refine_net = RefineNet().to(cfg.device)
refine_net.load_state_dict(torch.load(cfg.best_cnet, map_location=cfg.device), strict=False)
coarse_net.eval()
refine_net.eval()
refine_net.rhm_train = rh_model

bps_object = []
verts_object = []
for idx in idxs:
    frame_full_path = os.path.join(cfg.data_dir, frame_names[idx])
    frame_info = np.load(frame_full_path)
    bps_object.append(frame_info['bps_object'])
    verts_object.append(frame_info['verts_object'])

bps_object = torch.tensor(np.stack(bps_object), device=cfg.device, dtype=cfg.dtype)
verts_object = torch.tensor(np.stack(verts_object), device=cfg.device, dtype=cfg.dtype)
trans_rhand = torch.as_tensor(params['trans_rhand'][idxs]).to(cfg['device'])
global_orient_rhand_rotmat = torch.as_tensor(params['global_orient_rhand_rotmat'][idxs]).to(cfg['device'])
fpose_rhand_rotmat = torch.as_tensor(params['fpose_rhand_rotmat'][idxs]).to(cfg['device'])

input_ = {
    'bps_object': bps_object,
    'trans_rhand': trans_rhand,
    'global_orient_rhand_rotmat': global_orient_rhand_rotmat,
    'fpose_rhand_rotmat': fpose_rhand_rotmat,
}

out_params_cnet = coarse_net(**input_)
out_hand_verts_cnet = rh_model(**out_params_cnet).vertices

_, h2o, _ = point2point_signed(out_hand_verts_cnet, verts_object)
rnet_input = {}
rnet_input['trans_rhand_f'] = out_params_cnet['transl']
rnet_input['global_orient_rhand_rotmat_f'] = aa2rotmat(out_params_cnet['global_orient']).view(-1, 3, 3)
rnet_input['fpose_rhand_rotmat_f'] = aa2rotmat(out_params_cnet['hand_pose']).view(-1, 15, 3, 3)
rnet_input['verts_object'] = verts_object
rnet_input['h2o_dist']= h2o.abs()

out_params_rnet = refine_net(**rnet_input)
out_hand_verts_rnet = rh_model(**out_params_rnet).vertices

for i in range(out_hand_verts_cnet.shape[0]):
    hand_mesh_cnet = trimesh.Trimesh(vertices=out_hand_verts_cnet[i].detach().cpu().numpy(), faces=rh_model.faces)
    _ = hand_mesh_cnet.export(os.path.join(cfg.out_dir, 'recon', f"{idxs[i]:04d}_cnet.obj"))
    hand_mesh_rnet = trimesh.Trimesh(vertices=out_hand_verts_rnet[i].detach().cpu().numpy(), faces=rh_model.faces)
    _ = hand_mesh_rnet.export(os.path.join(cfg.out_dir, 'recon', f"{idxs[i]:04d}_rnet.obj"))

