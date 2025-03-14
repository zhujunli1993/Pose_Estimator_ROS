from __future__ import print_function

import absl.flags as flags

# datasets
flags.DEFINE_integer('obj_c', 6, 'number of categories')
flags.DEFINE_string('dataset', 'Real', 'CAMERA or CAMERA+Real')
flags.DEFINE_string('dataset_dir', './data/NOCS', 'path to the dataset')
flags.DEFINE_string('detection_dir', './data/segmentation_results', 'path to detection results')
flags.DEFINE_string('per_obj', '', 'only train an specified object')
flags.DEFINE_string('pretrained_clip_rot_model_path','','path of clip rotation')
flags.DEFINE_string('pretrained_clip_t_model_path','','path of clip trans')
flags.DEFINE_string('detection_pth','','path of ROS detection')
flags.DEFINE_string('depth_pth','','path of ROS depth')
flags.DEFINE_string('label_pth','','path of ROS label')
flags.DEFINE_string('bbox_pth','','path of ROS bbox')
flags.DEFINE_string('ROS_RGB_pth','','path of ROS RGB image')
flags.DEFINE_string('ROS_save_result_name','','name of ROS result RGB image')
# contrastive learning settings

flags.DEFINE_float('smooth_l1_beta',1.0, 'smooth l1')
flags.DEFINE_string('pointnet2_params','lighter', 'pointnet parameters: light, lighter, dense')
flags.DEFINE_integer('heads', 2, 'number of heads')
# dynamic zoom in
flags.DEFINE_float('DZI_PAD_SCALE', 1.5, '')
flags.DEFINE_string('DZI_TYPE', 'uniform', '')
flags.DEFINE_float('DZI_SCALE_RATIO', 0.25, '')
flags.DEFINE_float('DZI_SHIFT_RATIO', 0.25, '')

# input parameters
flags.DEFINE_integer("img_size", 256, 'size of the cropped image')
# pose settings
flags.DEFINE_string('pose_mode', 'original', 'the rotation representation')
# data aug parameters
flags.DEFINE_integer('roi_mask_r', 3, 'radius for mask aug')
flags.DEFINE_float('roi_mask_pro', 0.5, 'probability to augment mask')
flags.DEFINE_float('aug_pc_pro', 0.2, 'probability to augment pc')
flags.DEFINE_float('aug_pc_r', 0.2, 'max change 20% of the point')
flags.DEFINE_float('aug_rt_pro', 0.3, 'probability to augment rt')
flags.DEFINE_float('aug_bb_pro', 0.3, 'probability to augment size')
flags.DEFINE_float('aug_bc_pro', 0.3, 'box cage based augmentation, only valid for bowl, mug')


flags.DEFINE_integer('face_recon_c', 6 * 5, 'for every point, we predict its distance and normal to each face')
#  the storage form is 6*3 normal, then the following 6 parametes distance, the last 6 parameters confidence
flags.DEFINE_integer('gcn_sup_num', 7, 'support number for gcn')
flags.DEFINE_integer('gcn_n_num', 20, 'neighbor number for RF-F and ORL')

# point selection
flags.DEFINE_integer('random_points', 1024, 'number of points selected randomly')
flags.DEFINE_string('sample_method', 'basic', 'basic')

# train parameters
# train##################################################
flags.DEFINE_integer("cpu", 16, "cpu usage")
flags.DEFINE_integer("train", 1, "1 for train mode")
flags.DEFINE_string("train_stage", 'PoseNet_only', "for train stage")
# flags.DEFINE_integer('eval', 0, '1 for eval mode')
flags.DEFINE_string('device', 'cuda:0', '')
# flags.DEFINE_string("train_gpu", '0', "gpu no. for training")
flags.DEFINE_integer("num_workers", 20, "cpu cores for loading dataset")
flags.DEFINE_integer("seed", -1, "random seed for reproducibility")
flags.DEFINE_integer('batch_size', 16, '')
flags.DEFINE_integer('total_epoch', 150, 'total epoches in training')
flags.DEFINE_integer('train_steps', 1500, 'number of batches in each epoch')  # batchsize is 8, then 3000
#####################space is not enough, trade time for space####################
flags.DEFINE_integer('accumulate', 1, '')   # the real batch size is batchsize x accumulate

# test parameters

# for different losses
flags.DEFINE_string('fsnet_loss_type', 'l1', 'l1 or smoothl1')

flags.DEFINE_float('rot_1_w', 8.0, '')
flags.DEFINE_float('rot_2_w', 8.0, '')
flags.DEFINE_float('rot_regular', 4.0, '')
flags.DEFINE_float('tran_w', 8.0, '')
flags.DEFINE_float('size_w', 8.0, '')
flags.DEFINE_float('recon_w', 8.0, '')
flags.DEFINE_float('r_con_w', 1.0, '')

flags.DEFINE_float('recon_n_w', 3.0, 'normal estimation loss')
flags.DEFINE_float('recon_d_w', 3.0, 'dis estimation loss')
flags.DEFINE_float('recon_v_w', 1.0, 'voting loss weight')
flags.DEFINE_float('recon_s_w', 0.3, 'point sampling loss weight, important')
flags.DEFINE_float('recon_f_w', 1.0, 'confidence loss')
flags.DEFINE_float('recon_bb_r_w', 1.0, 'bbox r loss')
flags.DEFINE_float('recon_bb_t_w', 1.0, 'bbox t loss')
flags.DEFINE_float('recon_bb_s_w', 1.0, 'bbox s loss')
flags.DEFINE_float('recon_bb_self_w', 1.0, 'bb self')


flags.DEFINE_float('mask_w', 1.0, 'obj_mask_loss')

flags.DEFINE_float('geo_p_w', 1.0, 'geo point mathcing loss')
flags.DEFINE_float('geo_s_w', 10.0, 'geo symmetry loss')
flags.DEFINE_float('geo_f_w', 0.1, 'geo face loss, face must be consistent with the point cloud')

flags.DEFINE_float('prop_pm_w', 2.0, '')
flags.DEFINE_float('prop_sym_w', 1.0, 'importtannt for symmetric objects, can do point aug along reflection plane')
flags.DEFINE_float('prop_r_reg_w', 1.0, 'rot confidence must be sum to 1')
# training parameters
# learning rate scheduler
flags.DEFINE_float('lr', 1e-4, '')
 # initial learning rate w.r.t basic lr
flags.DEFINE_float('lr_pose', 1.0, '')
flags.DEFINE_integer('lr_decay_iters', 50, '')  # some parameter for the scheduler
### optimizer  ####
flags.DEFINE_string('lr_scheduler_name', 'flat_and_anneal', 'linear/warm_flat_anneal/')
flags.DEFINE_string('anneal_method', 'cosine', '')
flags.DEFINE_float('anneal_point', 0.72, '')
flags.DEFINE_string('optimizer_type', 'Ranger', '')
flags.DEFINE_float('weight_decay', 0.0, '')
flags.DEFINE_float('warmup_factor', 0.001, '')
flags.DEFINE_integer('warmup_iters', 1000, '')
flags.DEFINE_string('warmup_method', 'linear', '')
flags.DEFINE_float('gamma', 0.1, '')
flags.DEFINE_float('poly_power', 0.9, '')

# save parameters
#model to save
flags.DEFINE_integer('save_every', 10, '')  # save models every 'save_every' epoch
flags.DEFINE_integer('log_every', 100, '')  # save log file every 100 iterations
flags.DEFINE_string('ROS_output', 'output/models/distr', 'path to save ROS results')
# resume
flags.DEFINE_integer('resume', 0, '1 for resume, 0 for training from the start')
flags.DEFINE_string('resume_model', './output/models/model_149.pth', 'path to the saved model')
flags.DEFINE_integer('resume_point', 0, 'the epoch to continue the training')


###################for evaluation#################
flags.DEFINE_integer('eval_seed', -1, 'evaluation seed for reproducibility')
flags.DEFINE_integer('eval_inference_only', 0, 'inference without evaluation')


    