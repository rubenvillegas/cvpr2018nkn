import os
import sys
sys.path.append("./outside-code")
import matplotlib
matplotlib.use("Agg")
import time
import numpy as np
import scipy.misc as sm
import tensorflow as tf
import BVH as BVH
import Animation
from argparse import ArgumentParser
from online_retargeting_model import EncoderDecoderGRU
from os import listdir, makedirs, system
from os.path import exists
from random import shuffle
from utils import get_minibatches_idx
from utils import numpy_gaussian_noise
from utils import get_floor

os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

def main(gpu, batch_size, alpha, gamma, omega, euler_ord, max_steps, min_steps,
         num_layer, gru_units, optim, mem_frac, keep_prob, learning_rate):

  prefix = "Online_Retargeting_Mixamo"

  for kk,vv in locals().iteritems():
    if (kk != "prefix" and kk != "mem_frac" and kk != "batch_size" and
        kk != "min_steps" and kk != "max_steps" and kk != "gpu"):
      prefix += "_"+kk+"="+str(vv)

  layers_units = []
  for i in range(num_layer):
    layers_units.append(gru_units)

  data_path = "./datasets/train/"
  alldata = []
  alloffset = []
  allskel = []
  allnames = []

  folders = [f for f in listdir(data_path) if not f.startswith(".") and
             not f.endswith("py") and not f.endswith(".npz")]
  for folder in folders:
    files = [f for f in listdir(data_path+folder) if not f.startswith(".") and
             f.endswith("_seq.npy")]
    for cfile in files:
      # Put the skels at the same height as the sequence
      positions = np.load(data_path+folder+"/"+cfile[:-8]+"_skel.npy")
      if positions.shape[0] >= min_steps:
        sequence = np.load(data_path+folder+"/"+cfile[:-8]+"_seq.npy")
        offset = sequence[:,-8:-4]
        sequence = np.reshape(sequence[:,:-8], [sequence.shape[0], -1, 3])
        positions[:,0,:] = sequence[:,0,:]
        alldata.append(sequence)
        alloffset.append(offset)
        allskel.append(positions)
        allnames.append(folder)

  trainseq = alldata
  trainskel = allskel
  trainoffset = alloffset

  print("Number of examples: "+str(len(trainseq)))
  tskel = []
  for tt in trainskel:
    tskel.append(tt[0:1])
  allframes_n_skel = np.concatenate(trainseq+tskel)
  min_root = allframes_n_skel[:,0:1].min(axis=0)
  max_root = allframes_n_skel[:,0:1].max(axis=0)
  data_mean = allframes_n_skel.mean(axis=0)[None,:]
  offset_mean = np.concatenate(trainoffset).mean(axis=0)[None,:]
  data_std = allframes_n_skel.std(axis=0)[None,:]
  offset_std = np.concatenate(trainoffset).std(axis=0)[None,:]

  np.save(data_path[:-6]+"mixamo_local_motion_mean.npy", data_mean)
  np.save(data_path[:-6]+"mixamo_local_motion_std.npy", data_std)
  data_std[data_std == 0] = 1
  np.save(data_path[:-6]+"mixamo_global_motion_mean.npy", offset_mean)
  np.save(data_path[:-6]+"mixamo_global_motion_std.npy", offset_std)

  n_joints = alldata[0].shape[-2]

  for i in xrange(len(trainseq)):
    trainseq[i] = (trainseq[i] - data_mean) / data_std
    trainoffset[i] = (trainoffset[i] - offset_mean) / offset_std
    trainskel[i] = (trainskel[i] - data_mean) / data_std

  models_dir = "../models/"+prefix
  logs_dir = "../logs/"+prefix

  parents = np.array([-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 0, 10, 11, 12, 3, 14, 15,
                      16, 3, 18, 19, 20])
  with tf.device("/gpu:%d"%gpu):
    gru = EncoderDecoderGRU(
        batch_size,
        alpha,
        gamma,
        omega,
        euler_ord,
        n_joints,
        layers_units,
        max_steps,
        data_mean,
        data_std,
        offset_mean,
        offset_std,
        parents,
        keep_prob,
        logs_dir,
        learning_rate,
        optim,
    )

  if not exists(models_dir):
    makedirs(models_dir)

  if not exists(logs_dir):
    makedirs(logs_dir)

  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=False,
                                        gpu_options=gpu_options)) as sess:

    sess.run(tf.global_variables_initializer())

    loaded, model_name = gru.load(sess, models_dir)
    if loaded:
      print("[*] Load SUCCESSFUL")
      step = int(model_name.split("-")[-1])
    else:
      print("[!] Starting from scratch ...")
      step = 0

    total_steps = 50000
    gru.saver = tf.train.Saver(max_to_keep=10)
    while step < total_steps:
      mini_batches = get_minibatches_idx(len(trainseq), batch_size,
                                         shuffle=True)
      for _, batchidx in mini_batches:
        start_time = time.time()
        if len(batchidx) == batch_size:

          if min_steps >= max_steps:
            steps = np.repeat(max_steps, batch_size)
          else:
            steps = np.random.randint(low=min_steps, high=max_steps+1,
                                      size=(batch_size,))

          seqA_batch = []
          offsetA_batch = []
          skelA_batch = []
          seqB_batch = []
          offsetB_batch = []
          skelB_batch = []
          mask_batch = np.zeros((batch_size, max_steps), dtype="float32")

          for b in xrange(batch_size):
            low = 0
            high = trainseq[batchidx[b]].shape[0]-max_steps
            if low >= high:
              stidx = 0
            else:
              stidx = np.random.randint(low=low, high=high)

            cseqA = trainseq[batchidx[b]][stidx:stidx+max_steps]
            mask_batch[b,:np.min([steps[b], cseqA.shape[0]])] = 1.0
            if cseqA.shape[0] < max_steps:
              cseqA = np.concatenate((cseqA, np.zeros((max_steps-cseqA.shape[0],
                                                       n_joints, 3))))

            coffsetA = trainoffset[batchidx[b]][stidx:stidx+max_steps]
            if coffsetA.shape[0] < max_steps:
              coffsetA = np.concatenate((
                  coffsetA, np.zeros((max_steps-coffsetA.shape[0], 4))
              ))

            cskelA = trainskel[batchidx[b]][stidx:stidx+max_steps]
            if cskelA.shape[0] < max_steps:
              cskelA = np.concatenate((
                  cskelA, np.zeros((max_steps-cskelA.shape[0], n_joints, 3))
              ))

            seqA_batch.append(cseqA)
            offsetA_batch.append(coffsetA)
            skelA_batch.append(cskelA)
            seqB_batch.append(cseqA)
            offsetB_batch.append(coffsetA)
            skelB_batch.append(cskelA)


          seqA_batch = np.array(seqA_batch).reshape((batch_size, max_steps,-1))
          offsetA_batch = np.array(offsetA_batch).reshape((
              batch_size, max_steps, -1
          ))
          seqA_batch = np.concatenate((seqA_batch, offsetA_batch), axis=-1)
          skelA_batch = np.array(skelA_batch).reshape((
              batch_size, max_steps, -1
          ))

          seqB_batch = np.array(seqB_batch).reshape((batch_size, max_steps,-1))
          offsetB_batch = np.array(offsetB_batch).reshape((
              batch_size, max_steps, -1
          ))
          seqB_batch = np.concatenate((seqB_batch, offsetB_batch), axis=-1)
          skelB_batch = np.array(skelB_batch).reshape((
              batch_size, max_steps, -1
          ))

          mid_time = time.time()

          l = gru.train(sess, seqA_batch, skelA_batch, seqB_batch, skelB_batch,
                         mask_batch, step)

          print("step=%d/%d, loss=%.5f, time=%.2f+%.2f"
                %(step, total_steps, l, mid_time-start_time,
                  time.time()-mid_time))

          if np.isnan(l):
            return

          if step >= 1000 and step % 1000 == 0:
            gru.save(sess, models_dir, step)

          step = step + 1

    gru.save(sess, models_dir, step)


if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--gpu", type=int, dest="gpu", required=True,
                      help="GPU device id")
  parser.add_argument("--batch_size", type=int, dest="batch_size",
                      default=16, help="Batch size for training")
  parser.add_argument("--alpha", type=float, dest="alpha",
                      default=90.0, help="Rotation angle threshold")
  parser.add_argument("--gamma", type=float, dest="gamma",
                      default=10.0, help="Twist loss weight")
  parser.add_argument("--omega", type=float, dest="omega",
                      default=10.0, help="Smoothness weight")
  parser.add_argument("--euler_ord", type=str, dest="euler_ord",
                      default="yzx", help="Euler rotation order")
  parser.add_argument("--max_steps", type=int, dest="max_steps",
                      default=60, help="Maximum number of steps in sequence")
  parser.add_argument("--min_steps", type=int, dest="min_steps",
                      default=60, help="Minimun number of steps in sequence")
  parser.add_argument("--num_layer", type=int, default=1, dest="num_layer",
                      help="Number of hidden layers for GRU")
  parser.add_argument("--gru_units", type=int, default=512, dest="gru_units",
                      help="Number of hidden units for GRU")
  parser.add_argument("--optim", type=str, default="rmsprop", dest="optim",
                      help="Optimizer for training")
  parser.add_argument("--mem_frac", type=float, dest="mem_frac", default=1.0,
                      help="GPU memory fraction to take up")
  parser.add_argument("--keep_prob", type=float, default=1.0, dest="keep_prob",
                      required=False, help="Keep probability for dropout")
  parser.add_argument("--learning_rate", type=float, default=0.0001,
                      dest="learning_rate", required=False,
                      help="Keep probability for dropout")
  args = parser.parse_args()
  main(**vars(args))

