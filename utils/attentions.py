import tensorflow as tf
import cv2
import numpy as np

def calculate_mask(model, data, mask_size=(128, 128)):
    """
    Calculate mask in a vision transformer model given samples inputs.
    
    References: 
        * https://arxiv.org/pdf/2005.00928.pdf
        * https://www.kaggle.com/piantic/vision-transformer-vit-visualize-attention-map
        * https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    """
    
    _, att = model(data, True)
    
    att_mat = tf.stack(att)
    
    # averaging over heads
    att_mat = tf.math.reduce_mean(att_mat, axis=2)
    
    residual_att = tf.eye(att_mat.shape[2])
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / tf.expand_dims(tf.math.reduce_sum(aug_att_mat, axis=-1), axis=-1)
    
    joint_attentions = []
    joint_attentions.append(aug_att_mat[0, :, :])
    
    grid_size = int(np.sqrt(aug_att_mat.shape[-1]))
    for n in range(1, aug_att_mat.shape[0]):
        joint_attentions.append(tf.linalg.matmul(aug_att_mat[n, :, :], joint_attentions[n-1]))
        
        
    v = joint_attentions[-1]
    masks = tf.reshape(v[:, 0, 1:], (-1, grid_size, grid_size)).numpy()
    
    numpy_masks = []
    for mask in masks:
        mask = cv2.resize(mask / mask.max(), (100, 100))[..., None]
        numpy_masks.append(mask)
        
        
    return numpy_masks
    
    