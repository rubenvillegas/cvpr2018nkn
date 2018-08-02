# Neural Kinematic Networks for Unsupervised Motion Retargetting

This is the code for the CVPR 2018 paper [Neural Kinematic Networks for Unsupervised Motion Retargetting](https://arxiv.org/pdf/1804.05653.pdf) by Ruben Villegas, Jimei Yang, Duygu Ceylan and Honglak Lee.

![](https://github.com/rubenvillegas/cvpr2018nkn/blob/master/gifs/gangnam_style.gif)


Please follow the instructions below to run the code:

## Requirements
Our method works with works with
* Linux
* NVIDIA Titan X GPU
* Tensorflow version 1.3.0

## Installing Dependencies (Anaconda installation is recommended)
* pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp27-none-linux_x86_64.whl

## Download and install blender version 2.79
**Download and install from:**  
* https://www.blender.org/download/

## Downloading Data
**Train data:**  
Firstly, create an account in the [Mixamo](https://www.mixamo.com) website. Next, refer to Apendix D in our [paper](https://arxiv.org/pdf/1804.05653.pdf), and download the corresponding fbx animation files for each character folder in ./datasets/train/. Once the fbx files have been downloaded, run the following ***blender script*** to convert them into BVH files:
```
blender -b -P ./datasets/fbx2bvh.py
```
Finally, preprocess the bvh files into npy files by running the following command:
```
python ./datasets/preprocess.py
```

**Test data (already preprocessed):**  
```
./datasets/download_test.sh
```

## Training
**NKN Autoencoder:**
```
CUDA_VISIBLE_DEVICES=GPU_ID python ./src/train_online_retargeting_mixamo.py --gpu=GPU_ID --min_steps=60 --max_steps=60 --gru_units=512 --num_layer=2 --learning_rate=0.0001 --keep_prob=0.9 --alpha=100 --gamma=10.0 --omega=0.01 --euler_ord=yzx --optim=adam
```

**NKN with Cycle Consistency:**
```
CUDA_VISIBLE_DEVICES=GPU_ID python ./src/train_online_retargeting_cycle_mixamo.py --gpu=GPU_ID --min_steps=60 --max_steps=60 --gru_units=512 --num_layer=2 --learning_rate=0.0001 --keep_prob=0.9 --alpha=100 --gamma=10.0 --omega=0.01 --euler_ord=yzx --optim=adam
```

**NKN with Adversarial Cycle Consistency:**
```
CUDA_VISIBLE_DEVICES=GPU_ID python ./src/train_online_retargeting_cycle_adv_mixamo.py --gpu=GPU_ID --min_steps=60 --max_steps=60 --gru_units=512 --num_layer=2 --learning_rate=0.0001 --keep_prob=0.9 --beta=0.001 --alpha=100 --gamma=10.0 --omega=0.01 --euler_ord=yzx --optim=adam --norm_type=batch_norm --d_arch=2 --margin=0.3 --d_rand
```

## Inference from above training (BVH files will be saved in ./results/blender_files)
**NKN Autoencoder:**
```
CUDA_VISIBLE_DEVICES=GPU_ID python src/test_online_retargeting_mixamo.py --gpu=GPU_ID --prefix=Online_Retargeting_Mixamo_gru_units=512_optim=adam_learning_rate=0.0001_num_layer=2_alpha=100.0_euler_ord=yzx_omega=0.01_keep_prob=0.9_gamma=10.0
```

**NKN with Cycle Consistency:**
```
CUDA_VISIBLE_DEVICES=GPU_ID python src/test_online_retargeting_mixamo.py --gpu=GPU_ID --prefix=Online_Retargeting_Mixamo_Cycle_gru_units=512_optim=adam_learning_rate=0.0001_num_layer=2_alpha=100.0_euler_ord=yzx_omega=0.01_keep_prob=0.9_gamma=10.0
```

**NKN with Adversarial Cycle Consistency:**
```
CUDA_VISIBLE_DEVICES=GPU_ID python src/test_online_retargeting_mixamo.py --gpu=GPU_ID --prefix=Online_Retargeting_Mixamo_Cycle_Adv_beta=0.001_gru_units=512_optim=adam_d_arch=2_learning_rate=0.0001_omega=0.01_norm_type=batch_norm_d_rand=True_num_layer=2_alpha=100.0_euler_ord=yzx_margin=0.3_keep_prob=0.9_gamma=10.0
```
**Inference with your trained models:**  
Simply change the --prefix input

## Evaluation
**Evaluate outputs**
```
python ./results/evaluate_mixamo.py
```
**Evaluate on your trained models**  
Please change the paths in **./results/evaluate_mixamo.py**  

**Results location**  
Results will be located in **./results/quantitative/result_tables_mixamo_online.txt**  

## Generating videos
**Download blender files with character skins**
```
./results/download.sh
```
**Generate videos**
```
blender -b -P ./results/make_videos.py
```

## Coming soon: From human 3D pose estimates to 3D characters
.............

## Citation                                                                      
                                                                                 
If you find this useful, please cite our work as follows:                        
```                                                                              
@InProceedings{Villegas_2018_CVPR,
  author = {Villegas, Ruben and Yang, Jimei and Ceylan, Duygu and Lee, Honglak},
  title = {Neural Kinematic Networks for Unsupervised Motion Retargetting},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2018}
}
```

Please contact "ruben.e.villegas@gmail.com" if you have any questions.

