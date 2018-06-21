import os
import sys
import ipdb
sys.path.append("./outside-code")
sys.path.append("./")
import BVH as BVH
import numpy as np
import scipy.ndimage.filters as filters
import Animation
from Quaternions import Quaternions
from Pivots import Pivots
from os import listdir, makedirs, system
from os.path import exists

data_paths = ["./datasets/train/"]

def get_skel(joints, parents):
  c_offsets = []
  for j in xrange(parents.shape[0]):
    if parents[j] != -1:
      c_offsets.append(joints[j,:] - joints[parents[j],:])
    else:
      c_offsets.append(joints[j,:])
  return np.stack(c_offsets, axis=0)

def softmax(x, **kw):
    softness = kw.pop("softness", 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)

def process(positions):

    """ Put on Floor """
    fid_l, fid_r = np.array([8,9]), np.array([12,13])
    foot_heights = np.minimum(positions[:,fid_l,1],
                              positions[:,fid_r,1]).min(axis=1)
    floor_height = softmin(foot_heights, softness=0.5, axis=0)

    positions[:,:,1] -= floor_height

    """ Add Reference Joint """
    trajectory_filterwidth = 3
    reference = positions[:,0]
    positions = np.concatenate([reference[:,np.newaxis], positions], axis=1)

    """ Get Foot Contacts """
    velfactor, heightfactor = np.array([0.15,0.15]), np.array([9.0, 6.0])

    feet_l_x = (positions[1:,fid_l,0] - positions[:-1,fid_l,0])**2
    feet_l_y = (positions[1:,fid_l,1] - positions[:-1,fid_l,1])**2
    feet_l_z = (positions[1:,fid_l,2] - positions[:-1,fid_l,2])**2
    feet_l_h = positions[:-1,fid_l,1]
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) &
               (feet_l_h < heightfactor)).astype(np.float)

    feet_r_x = (positions[1:,fid_r,0] - positions[:-1,fid_r,0])**2
    feet_r_y = (positions[1:,fid_r,1] - positions[:-1,fid_r,1])**2
    feet_r_z = (positions[1:,fid_r,2] - positions[:-1,fid_r,2])**2
    feet_r_h = positions[:-1,fid_r,1]
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) &
               (feet_r_h < heightfactor)).astype(np.float)

    """ Get Root Velocity """
    velocity = (positions[1:,0:1] - positions[:-1,0:1]).copy()

    """ Remove Translation """
    positions[:,:,0] = positions[:,:,0] - positions[:,:1,0]
    positions[1:,1:,1] = (positions[1:,1:,1] - (positions[1:,:1,1] - 
                          positions[:1,:1,1]))
    positions[:,:,2] = positions[:,:,2] - positions[:,:1,2]

    """ Get Forward Direction """
    # Original indices + 1 for added reference joint
    sdr_l, sdr_r, hip_l, hip_r = 15, 19, 7, 11
    across1 = positions[:,hip_l] - positions[:,hip_r]
    across0 = positions[:,sdr_l] - positions[:,sdr_r]
    across = across0 + across1
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]

    direction_filterwidth = 20
    forward = np.cross(across, np.array([[0,1,0]]))
    forward = filters.gaussian_filter1d(forward, direction_filterwidth,
                                        axis=0, mode="nearest")
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    """ Remove Y Rotation """
    target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
    rotation = Quaternions.between(forward, target)[:,np.newaxis]
    positions = rotation * positions

    """ Get Root Rotation """
    velocity = rotation[1:] * velocity
    rvelocity = Pivots.from_quaternions(rotation[1:] * -rotation[:-1]).ps

    """ Add Velocity, RVelocity, Foot Contacts to vector """
    positions = positions[:-1]
    positions = positions.reshape(len(positions), -1)
    positions = np.concatenate([positions, velocity[:,:,0]], axis=-1)
    positions = np.concatenate([positions, velocity[:,:,1]], axis=-1)
    positions = np.concatenate([positions, velocity[:,:,2]], axis=-1)
    positions = np.concatenate([positions, rvelocity], axis=-1)
    positions = np.concatenate([positions, feet_l, feet_r], axis=-1)

    return positions

""" This script generated the local/global motion decoupled data and
    stores it for later training """

joints_list = ["Spine", "Spine1", "Spine2", "Neck", "Head", "LeftUpLeg",
               "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg",
               "RightLeg", "RightFoot", "RightToeBase", "LeftShoulder",
               "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder",
               "RightArm", "RightForeArm", "RightHand"]

for data_path in data_paths:
  print("Processing "+data_path)
  folders = sorted([f for f in listdir(data_path) if not f.startswith(".") and
                    not f.endswith("py") and not f.endswith("npz")])

  for folder in folders:
    files = sorted([f for f in listdir(data_path+folder) if f.endswith(".bvh")])
    for cfile in files:
      print(data_path+folder+"/"+cfile)
 
      anim, _, _ = BVH.load(data_path+folder+"/"+cfile)
  
      bvh_file = open(data_path+folder+"/"+cfile).read().split("JOINT")
      bvh_joints = [f.split("\n")[0] for f in bvh_file[1:]]
      to_keep = [0]
      for jname in joints_list:
        for k in xrange(len(bvh_joints)):
          if jname == bvh_joints[k][-len(jname):]:
            to_keep.append(k+1)
            break
  
      anim.parents = anim.parents[to_keep]
      for i in xrange(1,len(anim.parents)):
        """ If joint not needed, connect to the previous joint """
        if anim.parents[i] not in to_keep:
          anim.parents[i] = anim.parents[i] - 1
        anim.parents[i] = to_keep.index(anim.parents[i])
  
      anim.positions = anim.positions[:,to_keep,:]
      anim.rotations.qs = anim.rotations.qs[:,to_keep,:]
      anim.orients.qs = anim.orients.qs[to_keep,:]
      if anim.positions.shape[0] > 1:
        joints = Animation.positions_global(anim)
        joints = np.concatenate([joints, joints[-1:]], axis=0)
        new_joints = process(joints)[:,3:]
        np.save(data_path+folder+"/"+cfile[:-4]+"_seq.npy", new_joints)
        anim.rotations.qs[...] = anim.orients.qs[None]
        tjoints = Animation.positions_global(anim)
        anim.positions[...] = get_skel(tjoints[0], anim.parents)[None]
        anim.positions[:,0,:] = new_joints[:,:3]
        np.save(data_path+folder+"/"+cfile[:-4]+"_skel.npy", anim.positions)
        print(anim.parents)
        print("Success.")

print("Done.")

