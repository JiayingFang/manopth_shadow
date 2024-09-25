from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d.art3d import juggle_axes
import numpy as np
import torch

from manopth.manolayer import ManoLayer

id_list = [38, 122, 92, 214, 79, 78, 239, 234, 118, 215, 108, 279, 117, 119, 120, 121]

def generate_random_hand(batch_size=1, ncomps=6, mano_root='mano/models'):
    nfull_comps = ncomps + 3  # Add global orientation dims to PCA
    random_pcapose = torch.rand(batch_size, nfull_comps)
    mano_layer = ManoLayer(mano_root=mano_root)
    verts, joints = mano_layer(random_pcapose)
    return {'verts': verts, 'joints': joints, 'faces': mano_layer.th_faces}


def display_hand(hand_info, mano_faces=None, ax=None, alpha=0.2, batch_idx=0, show=True):
    """
    Displays hand batch_idx in batch of hand_info, hand_info as returned by
    generate_random_hand
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][
        batch_idx]
    if mano_faces is None:
        ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
    else:
        mesh = Poly3DCollection(verts[mano_faces], alpha=alpha)
        face_color = (141 / 255, 184 / 255, 226 / 255)
        edge_color = (50 / 255, 50 / 255, 50 / 255)
        mesh.set_edgecolor(edge_color)
        mesh.set_facecolor(face_color)
        ax.add_collection3d(mesh)
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
    cam_equal_aspect_3d(ax, verts.numpy())
    if show:
        plt.show()

class Animation:
    def __init__(self, vert_collection, mano_faces=None, joint_root1=None):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.mano_faces = mano_faces
        self.vert_collection = vert_collection
        self.joint_root1 = joint_root1
    
    def display_hand(self, alpha=0.2, batch_idx=0):
        """
        Init hand
        """
        hand_info = self.vert_collection[0]
        verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][
            batch_idx]
        if self.mano_faces is None:
            self.ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], alpha=0.1)
        else:
            self.mesh = Poly3DCollection(verts[self.mano_faces], alpha=alpha)
            face_color = (141 / 255, 184 / 255, 226 / 255)
            # edge_color = (50 / 255, 50 / 255, 50 / 255)
            # self.mesh.set_edgecolor(edge_color)
            self.mesh.set_facecolor(face_color)
            self.ax.clear()
            self.ax.add_collection3d(self.mesh)
        joints_tip = torch.vstack((joints[4], joints[12]))
        self.scatter, = self.ax.plot(joints_tip[:, 0], joints_tip[:, 1], joints_tip[:, 2], linestyle="", marker="o", color='g', markersize=3)# self.ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color='r')
        root_joint = (joints[4]+joints[12])/2
        self.root_point, = self.ax.plot(root_joint[0], root_joint[1], root_joint[2], linestyle="", marker="o", color='r', markersize=3)
        cam_equal_aspect_3d(self.ax, verts.numpy())
        return self.mesh, self.scatter, self.root_point
        
    def display_hand_animate(self, i, alpha=0.2, batch_idx=0):
        """
        Update hand mesh and keypoints
        """
        hand_info = self.vert_collection[i]
        verts, joints = hand_info['verts'][batch_idx], hand_info['joints'][
            batch_idx]
       
        self.mesh.set_verts(verts[self.mano_faces])
        
        self.mesh._facecolors2d = self.mesh._facecolor3d
        self.mesh._edgecolors2d = self.mesh._edgecolor3d

        joints_tip = torch.vstack((joints[4], joints[12]))
        self.scatter.set_data(joints_tip[:, 0], joints_tip[:, 1])
        self.scatter.set_3d_properties(joints_tip[:, 2])
        root_joint = (joints[4]+joints[12])/2
        root_joint = root_joint.reshape(1,3)
        self.root_point.set_data(root_joint[:,0], root_joint[:,1])
        self.root_point.set_3d_properties(root_joint[:,2])
        
        return self.mesh, self.scatter, self.root_point


def cam_equal_aspect_3d(ax, verts, flip_x=False):
    """
    Centers view on cuboid containing hand and flips y and z axis
    and fixes azimuth
    """
    extents = np.stack([verts.min(0), verts.max(0)], axis=1)
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    if flip_x:
        ax.set_xlim(centers[0] + r, centers[0] - r)
    else:
        ax.set_xlim(centers[0] - r, centers[0] + r)
    # Invert y and z axis
    ax.set_ylim(centers[1] + r, centers[1] - r)
    ax.set_zlim(centers[2] + r, centers[2] - r)
