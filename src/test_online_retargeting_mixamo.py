import os
import cv2
import sys
sys.path.append("./outside-code")
import time
import socket

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
import tensorflow as tf
import scipy.misc as sm
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
import numpy as np
import scipy.io as sio
from os import listdir, makedirs, system
from argparse import ArgumentParser
from online_retargeting_model import EncoderDecoderGRU
from utils import load_testdata
from utils import put_in_world_bvh
from utils import get_orient_start

def main(gpu, prefix, mem_frac):
  is_test = True

  data_path = "./datasets/test/"
  min_steps = 120
  max_steps = 120
  (testseq, testoffset, testoutseq,
   testskel, from_names, to_names,
   tgtjoints, tgtanims, inpjoints,
   inpanims, gtanims) = load_testdata(min_steps, max_steps)

  data_mean = np.load(data_path[:-5]+"mixamo_local_motion_mean.npy")
  data_std = np.load(data_path[:-5]+"mixamo_local_motion_std.npy")
  offset_mean = np.load(data_path[:-5]+"mixamo_global_motion_mean.npy")
  offset_std = np.load(data_path[:-5]+"mixamo_global_motion_std.npy")
  data_std[data_std == 0] = 1

  for i in xrange(len(testseq)):
    testseq[i] = (testseq[i] - data_mean) / data_std
    testoffset[i] = (testoffset[i] - offset_mean) / offset_std
    testskel[i] = (testskel[i] - data_mean) / data_std

  num_layer = int(prefix.split("num_layer=")[1].split("_")[0])
  gru_units = int(prefix.split("gru_units=")[1].split("_")[0])
  keep_prob = 1.0

  n_joints = testskel[0].shape[-2]

  layers_units = []
  for i in range(num_layer):
    layers_units.append(gru_units)

  if is_test:
    results_dir = "./results/outputs/test/"+prefix
  else:
    results_dir = "./results/outputs/train/"+prefix
  models_dir  = "./models/"+prefix

  parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15,
                      16, 3, 18, 19, 20])

  with tf.device("/gpu:%d"%gpu):
    gru = EncoderDecoderGRU(1,
                            None,
                            None,
                            None,
                            None,
                            n_joints,
                            layers_units,
                            max_steps,
                            data_mean,
                            data_std,
                            offset_mean,
                            offset_std,
                            parents,
                            keep_prob,
                            None,
                            None,
                            None,
                            is_train=False)

  data_mean = data_mean.reshape((1,1,-1))
  data_std = data_std.reshape((1,1,-1))

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        gpu_options=gpu_options)) as sess:

    sess.run(tf.global_variables_initializer())

    if "paper_models" in models_dir:
      if "Adv" in prefix:
        best_model = "EncoderDecoderGRU.model-46000"
      elif "Cycle" in prefix:
        best_model = "EncoderDecoderGRU.model-47000"
      else:
        best_model = "EncoderDecoderGRU.model-43000"
    else:
      best_model = None # will pick last model

    loaded, model_name = gru.load(sess, models_dir, best_model)
    if loaded:
      print("[*] Load SUCCESS")
    else:
      print("[!] Load failed...")
      return

    for i in xrange(len(testseq)):

      print "Testing: "+str(i)+"/"+str(len(testseq))

      mask_batch = np.zeros((1, max_steps), dtype="float32")
      seqA_batch = testseq[i][:max_steps].reshape([1, max_steps, -1])
      offsetA_batch = testoffset[i][:max_steps].reshape([1, max_steps, -1])
      seqA_batch = np.concatenate((seqA_batch, offsetA_batch), axis=-1)
      skelB_batch = testskel[i][:max_steps].reshape([1, max_steps, -1])

      step = max_steps
      mask_batch[0,:step] = 1.0

      res_path = results_dir+"/{0:05d}".format(i)
      if not os.path.exists(results_dir):
        os.makedirs(results_dir)

      outputB, quatsB = gru.predict(sess, seqA_batch, skelB_batch, mask_batch)
      outputB[:,:step,:-4] = outputB[:,:step,:-4]*data_std+data_mean
      outputB[:,:step,-4:] = outputB[:,:step,-4:]*offset_std+offset_mean
      seqA_batch[:,:step,:-4] = seqA_batch[:,:step,:-4]*data_std+data_mean
      seqA_batch[:,:step,-4:] = seqA_batch[:,:step,-4:]*offset_std+offset_mean
      gt = testoutseq[i][None, :max_steps].copy()

      tjoints = np.reshape(skelB_batch*data_std+data_mean, [max_steps, -1, 3])
      bl_tjoints = tjoints.copy()
      tgtanim, tgtnames, tgtftime = tgtanims[i]
      gtanim, gtnames, gtftime = gtanims[i]
      inpanim, inpnames, inpftime = inpanims[i]

      tmp_gt = Animation.positions_global(gtanim)
      start_rots = get_orient_start(
          tmp_gt, tgtjoints[i][14], tgtjoints[i][18],
          tgtjoints[i][6], tgtjoints[i][10]
      )

      """Exclude angles in exclude_list as they will rotate non-existent
         children During training."""
      exclude_list = [5, 17, 21, 9, 13]
      canim_joints = []
      cquat_joints = []
      for l in xrange(len(tgtjoints[i])):
        if l not in exclude_list:
          canim_joints.append(tgtjoints[i][l])
          cquat_joints.append(l)

      outputB_bvh = outputB[0].copy()

      """Follow the same motion direction as the input and zero speeds
         that are zero in the input."""
      outputB_bvh[:,-4:] = outputB_bvh[:,-4:] * (
          np.sign(seqA_batch[0,:,-4:]) * np.sign(outputB[0,:,-4:])
      )
      outputB_bvh[:,-3][np.abs(seqA_batch[0,:,-3]) <= 1e-2] = 0.

      outputB_bvh[:,:3] = gtanim.positions[:1,0,:].copy()
      wjs, rots = put_in_world_bvh(outputB_bvh.copy(), start_rots)
      tjoints[:,0,:] = wjs[0,:,0].copy()

      cpy_bvh = seqA_batch[0].copy()
      cpy_bvh[:,:3] = gtanim.positions[:1,0,:].copy()
      bl_wjs, _ = put_in_world_bvh(cpy_bvh.copy(), start_rots)
      bl_tjoints[:,0,:] = bl_wjs[0,:,0].copy()

      cquat = quatsB[0][:,cquat_joints].copy()

      if "Big_Vegas" in from_names[i]:
        from_bvh = from_names[i].replace("Big_Vegas", "Vegas")
      else:
        from_bvh = from_names[i]

      if "Warrok_W_Kurniawan" in to_names[i]:
        to_bvh = to_names[i].replace("Warrok_W_Kurniawan", "Warrok")
      else:
        to_bvh = to_names[i]

      bvh_path = "./results/blender_files/"+to_bvh.split("_")[-1]
      if not os.path.exists(bvh_path):
        os.makedirs(bvh_path)

      bvh_path += "/{0:05d}".format(i)
      BVH.save(bvh_path+"_from="+from_bvh+"_to="+to_bvh+"_gt.bvh",
               gtanim, gtnames, gtftime)

      tgtanim.positions[:,tgtjoints[i]] = bl_tjoints.copy()
      tgtanim.offsets[tgtjoints[i][1:]] = bl_tjoints[0,1:]

      BVH.save(bvh_path+"_from="+from_bvh+"_to="+to_bvh+"_cpy.bvh",
               tgtanim, tgtnames, tgtftime)
      tgtanim.positions[:,tgtjoints[i]] = tjoints
      tgtanim.offsets[tgtjoints[i][1:]] = tjoints[0,1:]

      """World rotation of character (global rotation)"""
      cquat[:,0:1,:] = (rots * Quaternions(cquat[:,0:1,:])).qs
      tgtanim.rotations.qs[:,canim_joints] = cquat

      BVH.save(bvh_path+"_from="+from_bvh+"_to="+to_bvh+".bvh",
               tgtanim, tgtnames, tgtftime)
      BVH.save(bvh_path+"_from="+from_bvh+"_to="+to_bvh+"_inp.bvh",
               inpanim, inpnames, inpftime)

      np.savez(res_path+"_from="+from_names[i]+"_to="+to_names[i]+".npz",
               outputA_=outputB[:,:step],
               outputB_=outputB[:,:step],
               quatsB=quatsB[:,:step],
               input_=seqA_batch[:,:step],
               gt=gt)

  print "Done."

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--gpu", type=int, dest="gpu", required=True,
                      help="GPU device id")
  parser.add_argument("--prefix", type=str, dest="prefix", required=True,
                      help="Model to test")
  parser.add_argument("--mem_frac", type=float, dest="mem_frac", default=0.1,
                      help="GPU memory fraction to take up")
  args = parser.parse_args()
  main(**vars(args))

