import trimesh
import torch
import json
import pytorch3d
import pytorch3d.loss
import pytorch3d.ops
import numpy as np
import pdb


valid_frames = {'313': [0, 30],
        '315': [0, 30, 60, 90, 120, 150, 180, 210, 240, 300, 330, 360, 390],
        '377': [30, 90, 120],
        '386': [150, 180, 270],
        '387': [150],
        '390': [720],
        '392': [30],
        '393': [0, 60, 120, 150, 180, 210, 240],
        '394': [0, 30, 60, 90, 120, 150, 180, 210, 240, 270],
        }


def normal_consistency_vertex(pred_trimesh, gt_trimesh):
    """
    :param pred: predicted trimesh
    :param gt trimesh: GT mesh trimesh
    """
    pred_vertices = np.array(pred_trimesh.vertices)
    pred_normals = np.array(pred_trimesh.vertex_normals)

    gt_vertices = np.array(gt_trimesh.vertices)
    gt_normals = np.array(gt_trimesh.vertex_normals)

    pred_verts_torch = torch.from_numpy(pred_vertices).double().unsqueeze(0).cuda()
    gt_verts_torch = torch.from_numpy(gt_vertices).double().unsqueeze(0).cuda()

    knn_ret = pytorch3d.ops.knn_points(gt_verts_torch, pred_verts_torch)
    p_idx = knn_ret.idx.squeeze(-1).detach().cpu().numpy()

    pred_normals = pred_normals[p_idx, :]

    consistency = 1 - np.linalg.norm(pred_normals - gt_normals, axis=-1).mean()

    return consistency

def filter_mesh(mesh, a, b, d, subject, save_path=None):
    # Filter out potential floating blobs
    labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
    components, cnt = np.unique(labels, return_counts=True)

    face_mask = (labels == components[np.argmax(cnt)])
    valid_faces = np.array(mesh.faces)[face_mask, ...]
    n_vertices = len(mesh.vertices)
    vertex_mask = np.isin(np.arange(n_vertices), valid_faces)
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        print("update color index")
        mesh.visual.vertex_colors = mesh.visual.vertex_colors[vertex_mask]
    mesh.update_faces(face_mask)
    mesh.update_vertices(vertex_mask)
    if save_path is not None:
        nb.export(save_path)
    mesh.fix_normals() 
    if subject in ['313', '315']:
        mesh = mesh.slice_plane([0, 0, d], [a, b, 1.0])
    else:
        mesh = mesh.slice_plane([0, 0, d], [-a, -b, -1.0])

    return mesh

nb_losses = []
our_losses = []

nb_nc = []
our_nc = []

with open('ground_planes.json', 'r') as f:
    ground_planes = json.load(f)

subject = '386'

a, b, d = ground_planes[subject]

for idx in valid_frames[subject]:
    gt = trimesh.load('/work/courses/digital_human/1/zju/CoreView_{}/gt_mesh/000{:03d}/womask_sphere/meshes/00300000.ply'.format(subject, idx+1))
    gt = filter_mesh(gt, a, b, d, subject)

    nb = trimesh.load('./exp/zju_{}_mono-direct-mlp_field-ingp-shallow_mlp-default/test-view/renders/fuse_idx_{}_post.ply'.format(subject, idx//30))
    # nb = trimesh.load('tmp/2dgs_mesh/{}_gt_camera/fuse{}.ply'.format(subject,idx//30))
    nb = filter_mesh(nb, a, b, d, subject)

    # gt.export('tmp/gt.ply')


    # pdb.set_trace()

    # Normal consistency
    nb_nc.append(normal_consistency_vertex(nb, gt))

    gt_verts = torch.from_numpy(gt.vertices * 100).double().unsqueeze(0).cuda()
    nb_verts = torch.from_numpy(nb.vertices * 100).double().unsqueeze(0).cuda()

    nb_loss = pytorch3d.loss.chamfer_distance(nb_verts, gt_verts)
    nb_losses.append(nb_loss[0])

    # print (nb_loss, our_loss)
    # nb_loss = pytorch3d.ops.knn_points(gt_verts, nb_verts)
    # our_loss = pytorch3d.ops.knn_points(gt_verts, our_verts)
    # nb_loss_ = pytorch3d.ops.knn_points(nb_verts, gt_verts)
    # our_loss_ = pytorch3d.ops.knn_points(our_verts, gt_verts)

    # print (nb_loss_.dists.max(), our_loss_.dists.max())

    # nb_losses.append(nb_loss.dists.mean() + nb_loss_.dists.mean())
    # our_losses.append(our_loss.dists.mean() + our_loss_.dists.mean())

print (torch.stack(nb_losses, dim=0).mean() / 2.0)

print (np.mean(nb_nc))
