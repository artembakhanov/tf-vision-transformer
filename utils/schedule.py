import tensorflow as tf
import numpy as np
from const import DECAY_TYPES

class ScheduleWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    # adopted from here: 
    # https://github.com/google-research/vision_transformer/blob/a51dbfa231262ffd1e8e52c041324e6de644d99b/vit_jax/hyper.py#L20
    def __init__(self, base, end, total_steps, warmup_steps=10_000, decay_type="linear"):
        super(ScheduleWithWarmup, self).__init__()

        self.base = base
        self.end = end
        self.total_steps = total_steps
        if decay_type not in DECAY_TYPES:
            raise ValueError(f'Unknown lr type {decay_type}. Please use one of {DECAY_TYPES}.')
        self.decay_type = decay_type
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        lr = tf.broadcast_to(self.base, step.shape)
        
        
        progress = tf.cast((step - self.warmup_steps), dtype=tf.float32) / float(self.total_steps - self.warmup_steps)
        progress = tf.clip_by_value(progress, 0.0, 1.0)
        
        if self.decay_type == 'linear':
            lr = self.end + (lr - self.end) * (1.0 - progress)
        elif self.decay_type == 'cosine':
            lr = lr * 0.5 * (1.0 + np.cos(np.pi * progress))
            

        if self.warmup_steps:
            lr = lr * tf.math.minimum(1., tf.cast(step, dtype=tf.float32) / self.warmup_steps)
        
        return lr