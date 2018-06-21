import sys 
sys.path.append("./outside-code")
import random
import numpy as np
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.patheffects as pe
import Animation as Animation
import BVH as BVH
from matplotlib.animation import ArtistAnimation
from mpl_toolkits.mplot3d import Axes3D
from os import listdir, makedirs, system
from os.path import isdir
from Quaternions import Quaternions

def softmax(x, **kw):
  softness = kw.pop("softness", 1.0)
  maxi, mini = np.max(x, **kw), np.min(x, **kw)
  return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, **kw):
  return -softmax(-x, **kw)


def get_floor(sk_in):
  h_root = sk_in[0,0,1]
  return np.min([
      h_root + sk_in[0,10,1] + sk_in[0,11,1] + sk_in[0,12,1] + sk_in[0,13,1],
      h_root + sk_in[0,6,1] + sk_in[0,7,1] + sk_in[0,8,1] + sk_in[0,9,1]
  ])


def get_orient_start(reference, sdr_l, sdr_r, hip_l, hip_r):
  """ Get Forward Direction """
  across1 = reference[0:1,hip_l] - reference[0:1,hip_r]
  across0 = reference[0:1,sdr_l] - reference[0:1,sdr_r]
  across = across0 + across1
  across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]

  direction_filterwidth = 20
  forward = np.cross(across, np.array([[0,1,0]]))
  forward = filters.gaussian_filter1d(forward, direction_filterwidth,
                                      axis=0, mode="nearest")
  forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

  """ Add Y Rotation """
  target = np.array([[0,0,1]]).repeat(len(forward), axis=0)
  rotation = Quaternions.between(forward, target)[:,np.newaxis]
  return -rotation


def animation_plot(animations, savepath, parents, interval=33.33):
  for ai in range(len(animations)):
    anim = animations[ai][0].copy()

    joints = anim[:,:-4].copy()
    root_x = anim[:,-4].copy()
    root_y = anim[:,-3].copy()
    root_z = anim[:,-2].copy()
    root_r = anim[:,-1].copy()
    joints = joints.reshape((len(joints), -1, 3))
    joints[:,:,0] = joints[:,:,0] - joints[0:1,0:1,0]
    joints[:,:,2] = joints[:,:,2] - joints[0:1,0:1,2]

#    fid_l, fid_r = np.array([4,5]), np.array([9,10])
#    foot_heights = np.minimum(joints[:,fid_l,1], joints[:,fid_r,1]).min(axis=1)
#    floor_height = softmin(foot_heights, softness=0.5, axis=0)
#    joints[:,:,1] -= floor_height

    rotation = Quaternions.id(1)
    offsets = []
    translation = np.array([[0,0,0]])

    for i in range(len(joints)):
      joints[i,:,:] = rotation * joints[i]
      joints[i,:,0] = joints[i,:,0] + translation[0,0]
      joints[i,:,1] = joints[i,:,1] + translation[0,1]
      joints[i,:,2] = joints[i,:,2] + translation[0,2]
      rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0,1,0])) * rotation
      offsets.append(rotation * np.array([0,0,1]))
      translation = translation + rotation * np.array([root_x[i], root_y[i], root_z[i]])

    animations[ai] = joints
  scale = 8.25*((len(animations))/2.)

  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection="3d")
  ax.set_xlim3d(-scale*30, scale*30)
  ax.set_zlim3d(0, scale*60)
  ax.set_ylim3d(-scale*30, scale*30)
  ax.set_xticks([], [])
  ax.set_yticks([], [])
  ax.set_zticks([], [])
  ax.view_init(0, 90)
  ax.set_aspect("equal")

  plt.tight_layout()

  acolors = ["green", "red"]#list(sorted(colors.cnames.keys()))[::-1]
  lines = []

  for ai, anim in enumerate(animations):
    lines.append([plt.plot([0,0], [0,0], [0,0], color=acolors[ai],
        lw=2, path_effects=[pe.Stroke(linewidth=3, foreground="black"),
                            pe.Normal()])[0] for _ in range(anim.shape[1])])

  def animate(i):
    changed = []
    for ai in range(len(animations)):
      offset = 200*(ai-((len(animations))/2))
      for j in range(len(parents)):
        if parents[j] != -1:
          lines[ai][j].set_data(
              [animations[ai][i,j,0]+offset,
               animations[ai][i,parents[j],0]+offset],
              [-animations[ai][i,j,2],
               -animations[ai][i,parents[j],2]])
          lines[ai][j].set_3d_properties(
              [animations[ai][i,j,1],
               animations[ai][i,parents[j],1]])
      changed += lines

    return changed

  ani = animation.FuncAnimation(
      fig, animate, np.arange(len(animations[0])), interval=interval
  )
  ani.save(savepath, writer="ffmpeg", fps=30)
  plt.close()


def get_minibatches_idx(n, minibatch_size, shuffle=False):
  """ 
  Used to shuffle the dataset at each iteration.
  """
  idx_list = np.arange(n, dtype="int32")

  if shuffle:
    random.shuffle(idx_list)

  minibatches = []
  minibatch_start = 0 
  # Some examples in training data may not be considered in epoch
  for i in range(n // minibatch_size):
    minibatches.append(
        idx_list[minibatch_start:minibatch_start + minibatch_size]
    )
    minibatch_start += minibatch_size


  return zip(range(len(minibatches)), minibatches)


def load_testdata(min_steps, max_steps, is_h36m=False):
  data_path = "./datasets/test/"

  inlocal = []
  inglobal = []
  tgtdata = []
  inpjoints = []
  inpanims = []
  tgtjoints = []
  tgtanims = []
  tgtskels = []
  gtanims = []
  from_names = []
  to_names = []

  km_kc = {"known_motion/Kaya/":"known_character/Warrok_W_Kurniawan1/",
           "known_motion/Big_Vegas/":"known_character/Malcolm1/"}
  km_nc = {"known_motion/AJ/":"new_character/Mutant1/",
           "known_motion/Peasant_Man/":"new_character/Liam1/"}
  nm_kc = {"new_motion/Granny/":"known_character/Malcolm2/",
           "new_motion/Claire/":"known_character/Warrok_W_Kurniawan2/"}
  nm_nc = {"new_motion/Mutant/":"new_character/Liam2/",
           "new_motion/Claire/":"new_character/Mutant2/"}

  joints_list = ["Spine", "Spine1", "Spine2", "Neck", "Head", "LeftUpLeg",
                 "LeftLeg", "LeftFoot", "LeftToeBase", "RightUpLeg",
                 "RightLeg", "RightFoot", "RightToeBase", "LeftShoulder",
                 "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder",
                 "RightArm", "RightForeArm", "RightHand"]

  test_list = [km_kc, km_nc, nm_kc, nm_nc]
  count = 0
  for test_item in test_list:
    for inp, tgt in test_item.iteritems():
      files = sorted([f for f in listdir(data_path+inp)
                      if not f.startswith(".") and f.endswith("_seq.npy")])
      for cfile in files:
        # Put the skels at the same height as the sequence
        tgtskel = np.load(data_path+tgt+"/"+cfile[:-8]+"_skel.npy")
        inpskel = np.load(data_path+inp+"/"+cfile[:-8]+"_skel.npy")
        if tgtskel.shape[0] >= min_steps+1:
          if not ("Claire" in inp and "Warrok" in tgt):
            count += 1
          inpanim, inpnames, inpftime = BVH.load(
              data_path+inp+"/"+cfile[:-8]+".bvh"
          )

          tgtanim, tgtnames, tgtftime = BVH.load(
              data_path+tgt+"/"+cfile[:-8]+".bvh"
          )

          gtanim = tgtanim.copy()

          ibvh_file = open(
              data_path+inp+"/"+cfile[:-8]+".bvh"
          ).read().split("JOINT")
          ibvh_joints = [f.split("\n")[0].split(":")[-1].split(" ")[-1]
                         for f in ibvh_file[1:]]
          ito_keep = [0]
          for jname in joints_list:
            for k in xrange(len(ibvh_joints)):
              if jname == ibvh_joints[k][-len(jname):]:
                ito_keep.append(k+1)
                break

          tbvh_file = open(
              data_path+tgt+"/"+cfile[:-8]+".bvh"
          ).read().split("JOINT")
          tbvh_joints = [f.split("\n")[0].split(":")[-1].split(" ")[-1]
                         for f in tbvh_file[1:]]
          tto_keep = [0]
          for jname in joints_list:
            for k in xrange(len(tbvh_joints)):
              if jname == tbvh_joints[k][-len(jname):]:
                tto_keep.append(k+1)
                break

          tgtanim.rotations.qs[...] = tgtanim.orients.qs[None]
          if not is_h36m:
            """ Copy joints we don't predict """
            cinames = []
            for jname in inpnames:
              cinames.append(jname.split(":")[-1])
  
            ctnames = []
            for jname in tgtnames:
              ctnames.append(jname.split(":")[-1])
  
            for jname in cinames:
              if jname in ctnames:
                idxt = ctnames.index(jname)
                idxi = cinames.index(jname)
                tgtanim.rotations[:,idxt] = inpanim.rotations[:,idxi].copy()
  
            tgtanim.positions[:,0] = inpanim.positions[:,0].copy()

          inseq = np.load(
              data_path+inp+"/"+cfile[:-8]+"_seq.npy"
          )

          if inseq.shape[0] < min_steps:
            continue

          outseq = np.load(data_path+tgt+"/"+cfile[:-8]+"_seq.npy")

          """Subtract lowers point in first timestep for floor contact"""
          floor_diff = inseq[0,1:-8:3].min() - outseq[0,1:-8:3].min()
          outseq[:,1:-8:3] += floor_diff
          tgtskel[:,0,1] = outseq[:,1].copy()

          offset = inseq[:,-8:-4]
          inseq = np.reshape(inseq[:,:-8], [inseq.shape[0], -1, 3])
          num_samples = inseq.shape[0]//max_steps

          for s in xrange(num_samples):
            inpjoints.append(ito_keep)
            tgtjoints.append(tto_keep)
            inpanims.append([
                inpanim.copy()[s*max_steps:(s+1)*max_steps], inpnames, inpftime
            ])
            tgtanims.append([
                tgtanim.copy()[s*max_steps:(s+1)*max_steps], tgtnames, tgtftime
            ])
            gtanims.append([
                gtanim.copy()[s*max_steps:(s+1)*max_steps], tgtnames, tgtftime
            ])
            inlocal.append(inseq[s*max_steps:(s+1)*max_steps])
            inglobal.append(offset[s*max_steps:(s+1)*max_steps])
            tgtdata.append(outseq[s*max_steps:(s+1)*max_steps,:-4])
            tgtskels.append(tgtskel[s*max_steps:(s+1)*max_steps])
            from_names.append(inp.split("/")[0]+"_"+inp.split("/")[1])
            to_names.append(tgt.split("/")[0]+"_"+tgt.split("/")[1])

          if not inseq.shape[0] % max_steps == 0:
            inpjoints.append(ito_keep)
            tgtjoints.append(tto_keep)
            inpanims.append([
                inpanim.copy()[-max_steps:], inpnames, inpftime
            ])
            tgtanims.append([
                tgtanim.copy()[-max_steps:], tgtnames, tgtftime
            ])
            gtanims.append([
                gtanim.copy()[-max_steps:], tgtnames, tgtftime
            ])
            inlocal.append(inseq[-max_steps:])
            inglobal.append(offset[-max_steps:])
            tgtdata.append(outseq[-max_steps:,:-4])
            tgtskels.append(tgtskel[-max_steps:])
            from_names.append(inp.split("/")[0]+"_"+inp.split("/")[1])
            to_names.append(tgt.split("/")[0]+"_"+tgt.split("/")[1])

  return (inlocal, inglobal, tgtdata, tgtskels, from_names,
          to_names, tgtjoints, tgtanims, inpjoints, inpanims, gtanims)


def put_in_world(states):
  joints = states[:,:-4]
  root_x = states[:,-4]
  root_y = states[:,-3]
  root_z = states[:,-2]
  root_r = states[:,-1]

  joints = joints.reshape(joints.shape[:1] + (-1, 3))

  rotation = Quaternions.id(1)
  rotations = []
  offsets = []
  translation = np.array([[0,0,0]])

  for i in range(len(joints)):
    rotation = Quaternions.from_angle_axis(
        -root_r[i], np.array([0,1,0])
    ) * rotation
    rotations.append(rotation.qs[:,None,:])
    joints[i,:,:] = rotation * joints[i]
    joints[i,:,0] = joints[i,:,0] + translation[0,0]
    joints[i,:,1] = joints[i,:,1] + translation[0,1]
    joints[i,:,2] = joints[i,:,2] + translation[0,2]
    offsets.append(rotation * np.array([0,0,1]))
    translation = translation + rotation * np.array(
        [root_x[i], root_y[i], root_z[i]]
    )

  return joints[None], Quaternions(np.concatenate(rotations, axis=0))


def put_in_world_bvh(states, start_rots):
  joints = states[:,:-4]
  root_x = states[:,-4]
  root_y = states[:,-3]
  root_z = states[:,-2]
  root_r = states[:,-1]

  joints = joints.reshape(joints.shape[:1] + (-1, 3))

  rotation = start_rots[0] * Quaternions.id(1)
  rotations = []
  offsets = []
  translation = np.array([[0,0,0]])

  for i in range(len(joints)):

    joints[i,1:,:] = rotation * joints[i,1:,:]
    joints[i,:,0] = joints[i,:,0] + translation[0,0]
    joints[i,:,1] = joints[i,:,1] + translation[0,1]
    joints[i,:,2] = joints[i,:,2] + translation[0,2]

    rotations.append(rotation.qs[:,None,:])
    rotation = Quaternions.from_angle_axis(
        -root_r[i], np.array([0,1,0])
    ) * rotation

    offsets.append(rotation * np.array([0,0,1]))
    translation = translation + rotation * np.array(
        [root_x[i], root_y[i], root_z[i]]
    )

  return joints[None], Quaternions(np.concatenate(rotations, axis=0))


def put_in_world_h36m(states):
  joints = states[:,:-4]
  root_x = states[:,-4]
  root_y = states[:,-3]
  root_z = states[:,-2]
  root_r = states[:,-1]

  joints = joints.reshape(joints.shape[:1] + (-1, 3))
  joints[:,:,0] = joints[:,:,0] - joints[0:1,0:1,0]
  joints[:,:,2] = joints[:,:,2] - joints[0:1,0:1,2]

  rotation = Quaternions.id(1)
  rotations = []
  offsets = []
  translation = np.array([[0,0,0]])

  for i in range(len(joints)):
    rotation = Quaternions.from_angle_axis(
        -root_r[i], np.array([0,1,0])
    ) * rotation
    rotations.append(rotation.qs[:,None,:])
    joints[i,:,:] = rotation * joints[i]
    joints[i,:,0] = joints[i,:,0] + translation[0,0]
    joints[i,:,1] = joints[i,:,1] + translation[0,1]
    joints[i,:,2] = joints[i,:,2] + translation[0,2]
    offsets.append(rotation * np.array([0,0,1]))
    translation = translation + rotation * np.array(
        [root_x[i], root_y[i], root_z[i]]
    )

  return joints[None], Quaternions(np.concatenate(rotations, axis=0))


def numpy_gaussian_noise(input_, input_mean, input_std, scale):
  noise = np.random.normal(size=input_.shape, scale=scale)
  noisy_input = noise + input_*input_std + input_mean
  return (noisy_input - input_mean)/input_std

