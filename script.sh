export CUDA_VISIBLE_DEVICES=0
# Evaluate the trained model
python -m evaluation.evaluate_ros \
    --resume 1 --dataset 'Real' \
    --pretrained_clip_rot_model_path 'pretrained_pt/rot_contrast.pt' \
    --pretrained_clip_t_model_path 'pretrained_pt/trans_contrast.pt' \
    --resume_model 'pretrained_pt/pose_estimator.pt' \
    --depth_pth 'maskrcnn_data/4_depth.npy' \
    --detection_pth  'maskrcnn_data/4_masks.npy' \
    --label_pth 'maskrcnn_data/4_labels.npy' \
    --bbox_pth 'maskrcnn_data/4_boxes.npy' \
    --ROS_RGB_pth 'maskrcnn_data/4.jpg' \
    --ROS_output 'ROS_output' \
    --ROS_save_result_name '4'
