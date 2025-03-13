import os
import torch
import random
from network.Pose_Estimator import Pose_Estimator  
from contrast.Cont_split_trans import Model_Trans_all as Model_trans
from contrast.Cont_split_rot import Model_Rot_all as Model_rot
from tools.geom_utils import generate_RT
from config.config import flags

from absl import app
import sys
FLAGS = flags.FLAGS
from evaluation.load_data_eval_ros import PoseDataImg
import numpy as np
import time
import cv2
# from creating log
import tensorflow as tf
from evaluation.eval_utils_v1 import setup_logger
from evaluation.eval_utils_v1 import compute_degree_cm_mAP, draw_bbox
from tqdm import tqdm
import pickle

def draw_save_all(opt, result, save_dir, save_name):
    """ Load data and draw visualization results.
    """
    intrinsics = result['K']
    img_pth = result['image_path']
    
    image = cv2.imread(img_pth)
    
    # Get the correct RTs for Class_ids. If the target is missing we will return np.eye(). If multi-target is matched, we only keep the first.
    all_gt_RTs = []
    all_pred_RTs = []
    all_gt_scales = []
    all_pred_scales = []
    misses = []
    # all_pred_class = []
    result["cat_id"] = result["cat_id"].detach().cpu().numpy()
    for i in tqdm(range(len(result["cat_id"]))):
        
        
        pred_RTs = result["pred_RTs"][i]
        pred_scales = result["pred_scales"][i]
        
        gt_RTs = pred_RTs
        gt_scales = pred_scales
        miss = False
        if len(pred_RTs) <= 0:

            pred_RTs = np.eye(4)
            pred_RTs = np.broadcast_to(pred_RTs, gt_RTs.shape)
            pred_scales = np.zeros(gt_scales.shape)
        
        misses.append(miss)
        all_pred_RTs.append(pred_RTs)
        all_gt_RTs.append(gt_RTs)
        all_pred_scales.append(pred_scales)
        all_gt_scales.append(gt_scales)
        
        
        with open(os.path.join(save_dir, save_name+'_pred_RT.txt'), "a") as file_1:
            np.savetxt(file_1, pred_RTs)
        with open(os.path.join(save_dir, save_name+'_pred_scales.txt'), "a") as file_2:
            np.savetxt(file_2, pred_scales)
        with open(os.path.join(save_dir, save_name+'_pred_class.txt'), "a") as file_3:
            np.savetxt(file_3, np.array([result["cat_id"][i]]).astype(int), fmt='%i')
              
    (h, w) = image.shape[:2]
    center = (w/2, h/2)

    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))

    
    draw_bbox(image, all_gt_RTs, all_pred_RTs, all_gt_scales, all_pred_scales, class_ids=result["cat_id"], misses=misses, 
              intrinsics=intrinsics, save_path=os.path.join(save_dir, save_name+'.png'))
    
    
def seed_init_fn(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return

device = 'cuda'

def evaluate(argv):
    if FLAGS.eval_seed == -1:
        seed = int(time.time())
    else:
        seed = FLAGS.eval_seed
    seed_init_fn(seed)

    tf.compat.v1.disable_eager_execution()

    
    Train_stage = 'PoseNet_only'

    t_inference = 0.0
    img_count = 0
    
    
    network = Pose_Estimator(Train_stage)
    network = network.to(device)
    ################ Load CLIP Models #########################
    ''' Load pretrained CLIP trianing agent'''
    clip_model_rot = Model_rot().to(FLAGS.device)
    if FLAGS.pretrained_clip_rot_model_path:
        clip_model_rot.load_state_dict(torch.load(FLAGS.pretrained_clip_rot_model_path))
    else:
        print("No Pretrained Rotation CLIP Model !!!")
        sys.exit()
        
    clip_model_t = Model_trans().to(FLAGS.device)
    if FLAGS.pretrained_clip_t_model_path:
        clip_model_t.load_state_dict(torch.load(FLAGS.pretrained_clip_t_model_path))
    else:
        print("No Pretrained Translation CLIP Model !!!")
        sys.exit()
    # Freeze clip model parameters
    for param in clip_model_t.parameters():
        param.requires_grad = False     
    for param in clip_model_rot.parameters():
        param.requires_grad = False   
    ################ Finish Loading CLIP Models #########################  
    
    #################### Loading Pose Estimator ############################
    if FLAGS.resume:
        state_dict = torch.load(FLAGS.resume_model)['posenet_state_dict']
        unnecessary_nets = ['posenet.face_recon.conv1d_block', 'posenet.face_recon.face_head', 'posenet.face_recon.recon_head']
        for key in list(state_dict.keys()):
            for net_to_delete in unnecessary_nets:
                if key.startswith(net_to_delete):
                    state_dict.pop(key)
            # Adapt weight name to match old code version. 
            # Not necessary for weights trained using newest code. 
            # Dose not change any function. 
            if 'resconv' in key:
                state_dict[key.replace("resconv", "STE_layer")] = state_dict.pop(key)
        network.load_state_dict(state_dict, strict=True) 
    else:
        raise NotImplementedError
    
    # start to test
    network = network.eval()
    clip_model_rot = clip_model_rot.eval()
    clip_model_t = clip_model_t.eval()
    pred_results = []
    
    ######################## Intrinsic Parameters #############################
    c_X = 324.8378935141304
    c_Y = 235.4308556036733
    f_X = 527.8716747158403
    f_Y = 520.6781086489601
    
    intrinsic = np.array(([f_X, 0.0, c_X], [0.0, f_Y, c_Y], [0.0, 0.0, 1.0]), dtype=float)
    
    ######################## Loading Data #############################
    data = PoseDataImg(FLAGS.depth_pth,FLAGS.detection_pth,FLAGS.label_pth,FLAGS.bbox_pth,intrinsic,pc_pth=None)
    mean_shape = data['mean_shape'].to(device)
    sym = data['sym_info'].to(device)
    
    ########################### Start Inference #########################
    t_start = time.time()
    output_dict \
        = network(clip_r_func=clip_model_rot.to(device),
                    clip_t_func=clip_model_t.to(device),
                    PC=data['pcl_in'].to(device), 
                    obj_id=data['cat_id_0base'].to(device), 
                    mean_shape=mean_shape,
                    sym=sym,
                #   def_mask=data['roi_mask'].to(device)
                    )
    t_inference += time.time() - t_start
    img_count += 1
    ########################### End Inference #########################
    
    
    p_green_R_vec = output_dict['p_green_R'].detach()
    p_red_R_vec = output_dict['p_red_R'].detach()
    p_T = output_dict['Pred_T'].detach()
    p_s = output_dict['Pred_s'].detach()
    f_green_R = output_dict['f_green_R'].detach()
    f_red_R = output_dict['f_red_R'].detach()
    pred_s = p_s + mean_shape
    pred_RT = generate_RT([p_green_R_vec, p_red_R_vec], [f_green_R, f_red_R], p_T, mode='vec', sym=sym)

    
    
    if pred_RT is not None:
        pred_RT = pred_RT.detach().cpu().numpy()
        pred_s = pred_s.detach().cpu().numpy()
        data['pred_RTs'] = pred_RT
        data['pred_scales'] = pred_s
        data['image_path'] = FLAGS.ROS_RGB_pth
        data['K'] = intrinsic
    else:
        assert NotImplementedError
    pred_results.append(data)
    
    torch.cuda.empty_cache()
    
    print('inference time:', t_inference / img_count)
    
        
    ################### Draw Visualization ##############################    
    
    os.makedirs(FLAGS.ROS_output, exist_ok=True)
    
    
    draw_save_all(FLAGS, pred_results[0], FLAGS.ROS_output, FLAGS.ROS_save_result_name)
    
    print("Drawing Done!")
    print("Output Visualization Is Saved In "+FLAGS.ROS_output+"/"+FLAGS.ROS_save_result_name+'.png !!')
    print("Predicted RT Is Saved In "+FLAGS.ROS_output+"/"+FLAGS.ROS_save_result_name+'_pred_RT.txt !!')
    print("Predicted Scale Is Saved In "+FLAGS.ROS_output+"/"+FLAGS.ROS_save_result_name+'_pred_scales.txt !!')
    print("Predicted and Valid Class Label Is Saved In "+FLAGS.ROS_output+"/"+FLAGS.ROS_save_result_name+'_pred_class.txt !!')
if __name__ == "__main__":
    app.run(evaluate)
