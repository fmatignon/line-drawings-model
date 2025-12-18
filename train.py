"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""

import time
import os
import torch
import numpy as np
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp, tensor2im, save_image


if __name__ == "__main__":
    opt = TrainOptions().parse()  # get training options
    opt.device = init_ddp()
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    print(f"The number of training images = {dataset_size}")

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()
        # Set epoch for DistributedSampler
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                visualizer.plot_current_losses(total_iters, losses)

            if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        model.update_learning_rate()  # update learning rates at the end of every epoch

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
            model.save_networks("latest")
            model.save_networks(epoch)

        print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {time.time() - epoch_start_time:.0f} sec")

    # After training completes, process all training images for visual verification
    print("\n" + "="*50)
    print("Processing all training images for visual verification...")
    print("="*50)
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory for training results
    results_dir = os.path.join(opt.checkpoints_dir, opt.name, "training_results")
    os.makedirs(results_dir, exist_ok=True)
    print(f"Saving results to: {results_dir}")
    
    # Process all training images
    with torch.no_grad():
        for i, data in enumerate(dataset):
            model.set_input(data)
            model.test()  # This calls forward() and compute_visuals()
            
            # Get the generated images
            visuals = model.get_current_visuals()
            
            # Get image path to extract filename
            image_path = model.get_image_paths()[0]
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            
            # Get all three images: input (real_A), output (fake_B), and ground truth (real_B)
            images_to_combine = []
            
            if "real_A" in visuals:
                real_A_tensor = visuals["real_A"]
                # Extract only RGB channels (first 3) if real_A has 4 channels (RGB + Canny)
                if real_A_tensor.shape[1] == 4:
                    real_A_tensor = real_A_tensor[:, :3, :, :]  # Take only first 3 channels
                real_A_img = tensor2im(real_A_tensor)
                images_to_combine.append(real_A_img)
            
            if "fake_B" in visuals:
                fake_B_img = tensor2im(visuals["fake_B"])
                images_to_combine.append(fake_B_img)
            
            if "real_B" in visuals:
                real_B_img = tensor2im(visuals["real_B"])
                images_to_combine.append(real_B_img)
            
            # Concatenate images horizontally: original | result | target
            if len(images_to_combine) == 3:
                combined_img = np.concatenate(images_to_combine, axis=1)
                save_image(combined_img, os.path.join(results_dir, f"{name}_comparison{ext}"))
            else:
                print(f"Warning: Expected 3 images but got {len(images_to_combine)} for {filename}")
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{dataset_size} images...")
    
    print(f"\nCompleted! Processed {dataset_size} training images.")
    print(f"Results saved to: {results_dir}")
    print("="*50 + "\n")

    cleanup_ddp()
