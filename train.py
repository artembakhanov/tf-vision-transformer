from argparse import ArgumentParser
from const import DECAY_TYPES, DATASETS, CLASSES_NUM
import numpy as np
from utils.schedule import ScheduleWithWarmup
from models import VisionTransformer
from utils.datasets import load_data
import tensorflow as tf
import datetime
from coolname import generate_slug
import logging
from pathlib import Path
import sys


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--logdir", default="logs/train/")
    parser.add_argument("--dataset", default="mnist", choices=DATASETS, help="dataset to train on")
    parser.add_argument("--patch-size", default=4, type=int, help="side size of a patch")
    parser.add_argument("--latent-dim", default=32, type=int, help="size of latent vectors in encoder layers")
    parser.add_argument("--heads-num", default=4, type=int, help="number of heads in self-attention layer")
    parser.add_argument("--mlp-dim", default=64, type=int, help="size of hidden layer in encoder MLP")
    parser.add_argument("--encoders_num", default=4, type=int, help="number of encoders")
    parser.add_argument("--dropout-rate", default=0.1, type=float, help="dropout rate")
    parser.add_argument("--base-lr", default=1e-3, type=float)
    parser.add_argument("--end-lr", default=1e-3, type=float)
    parser.add_argument("--warmup-steps", default=0, type=int, help="number of steps for lr warmup")
    parser.add_argument("--decay-type", default="linear", choices=DECAY_TYPES)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=40, type=int)
    parser.add_argument("--model-name", default=generate_slug(2))
    parser.add_argument("--checkpoints", action='store_true')
    parser.add_argument("--checkpoints-dir", default="checkpoints/")
    parser.add_argument("--save-freq", default=3, type=int)
    parser.add_argument("--model-dir", default="saved_models/")
    args = parser.parse_args()
    
    model_time_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{args.model_name}"
    log_dir = Path(args.logdir) / model_time_name
    checkpoints_dir = Path(args.checkpoints_dir)
    
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir) / model_time_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_filepath = checkpoints_dir / model_time_name / "weights.{epoch:03d}-{val_loss:.2f}-{val_accuracy:.2f}.ckpt"
    
    # setup logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    
    output_file_handler = logging.FileHandler(log_dir / "run.log")
    stdout_handler = logging.StreamHandler(sys.stdout)

    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    
    
    logger.info(f"Starting training model {model_time_name}.")
    logger.info(f"Logs will be saved here: {Path.cwd() / log_dir}")
    if args.checkpoints:
        logger.info(f"Checkpoints will be saved in {Path.cwd() / checkpoints_dir / model_time_name}")
    logger.info(f"Model weights will be saved in {model_dir}")
    
    logger.info(f"")
    
    
    logger.info(f"Starting {args.dataset} dataset loading.")
    train_ds, test_ds = load_data(args.dataset, args.batch_size)
    logger.info(f"{args.dataset} dataset has been loaded.")
    
    
    model = VisionTransformer(patch_size=args.patch_size, 
                          latent_dim=args.latent_dim, 
                          heads_num=args.heads_num, 
                          mlp_dim=args.mlp_dim, 
                          encoders_num=args.encoders_num, 
                          mlp_head_dim=None, 
                          classes_num=CLASSES_NUM[args.dataset], 
                          dropout_rate=args.dropout_rate)
    
    learning_rate = ScheduleWithWarmup(
        base=args.base_lr,
        end=args.end_lr,
        total_steps=len(train_ds) * args.epochs,
        warmup_steps=args.warmup_steps,
        decay_type=args.decay_type,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
              ),
              metrics=['accuracy'])
    
    for x, _ in train_ds:
        _ = model(x, True)
        break
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    callbacks = [tensorboard_callback]
    
    if args.checkpoints:
        checkpoints_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_freq=args.save_freq,
            save_best_only=True
        )
        
        callbacks.append(checkpoints_callback)

    logger.info(f"Starting training.")
    model.fit(x=train_ds, 
              batch_size=args.batch_size,
              epochs=args.epochs, 
              validation_data=test_ds, 
              callbacks=callbacks)
    logger.info(f"Training finished.")
    
    model.save(model_dir / "final.model")
    logger.info(f"Model saved to {model_dir / 'final.model'}.")

    model.save_weights(model_dir / 'final.weights')
    logger.info(f"Model weights saved to {model_dir / 'final.weights'}.")


    
