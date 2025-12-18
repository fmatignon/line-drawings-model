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
import cv2
import torchvision.transforms as transforms
from PIL import Image as PILImage
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp, tensor2im
from data.base_dataset import get_transform


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
    
    # Get the list of images that were actually used during training
    # The dataset is a DataLoader, access the underlying dataset via .dataset
    underlying_dataset = dataset.dataset
    training_image_paths = []
    for i in range(len(underlying_dataset)):
        data = underlying_dataset[i]
        image_path = data.get("A_paths", None) or data.get("B_paths", None)
        if image_path:
            training_image_paths.append(image_path)
    
    print(f"Found {len(training_image_paths)} images that were used during training")
    
    # Extract base directory paths
    base_dir = opt.dataroot.replace(opt.phase, "").rstrip("/")
    if not base_dir:
        base_dir = os.path.dirname(opt.dataroot.rstrip("/"))
    
    originals_dir = os.path.join(base_dir, "originals")
    if not os.path.exists(originals_dir):
        # Try alternative path
        originals_dir = "dataset/originals"
    
    line_drawings_dir = os.path.join(base_dir, "line_drawings")
    if not os.path.exists(line_drawings_dir):
        # Try alternative path
        line_drawings_dir = "dataset/line_drawings"
    
    # Helper function to resize with padding (same as preprocess_data.py)
    def resize_with_padding(img, size=512, interpolation=cv2.INTER_AREA):
        """Resize image to size x size maintaining aspect ratio with padding."""
        h, w = img.shape[:2]
        scale = min(size / w, size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
        
        if len(img.shape) == 3:
            canvas = np.zeros((size, size, 3), dtype=img.dtype)
        else:
            canvas = np.zeros((size, size), dtype=img.dtype)
        
        paste_x = (size - new_w) // 2
        paste_y = (size - new_h) // 2
        
        if len(img.shape) == 3:
            canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized
        else:
            canvas[paste_y : paste_y + new_h, paste_x : paste_x + new_w] = resized
        
        return canvas
    
    # Process only images that were used during training
    with torch.no_grad():
        for i, aligned_image_path in enumerate(training_image_paths):
            # Extract filename from the aligned image path
            # The aligned images are in dataset/aligned or dataset/train, etc.
            aligned_filename = os.path.basename(aligned_image_path)
            name, ext = os.path.splitext(aligned_filename)
            
            # Find corresponding original and line drawing files
            original_path = os.path.join(originals_dir, aligned_filename)
            line_drawing_path = os.path.join(line_drawings_dir, aligned_filename)
            
            # If exact match not found, try without extension variations
            if not os.path.exists(original_path):
                # Try to find with different extensions
                found = False
                for ext_try in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                    try_path = os.path.join(originals_dir, name + ext_try)
                    if os.path.exists(try_path):
                        original_path = try_path
                        found = True
                        break
                if not found:
                    print(f"Skipping {aligned_filename}: Original image not found in {originals_dir}")
                    continue
            
            if not os.path.exists(line_drawing_path):
                # Try to find with different extensions
                found = False
                for ext_try in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
                    try_path = os.path.join(line_drawings_dir, name + ext_try)
                    if os.path.exists(try_path):
                        line_drawing_path = try_path
                        found = True
                        break
                if not found:
                    print(f"Skipping {aligned_filename}: Line drawing not found in {line_drawings_dir}")
                    continue
            
            # Load and preprocess original image (same as preprocess_data.py)
            img_a = cv2.imread(original_path)
            if img_a is None:
                print(f"Skipping {filename}: Failed to read original image.")
                continue
            
            # Resize with padding to 512x512 (same as preprocess_data.py)
            img_a_padded = resize_with_padding(img_a, size=opt.crop_size, interpolation=cv2.INTER_AREA)
            
            # Convert to PIL Image
            img_a_pil = PILImage.fromarray(cv2.cvtColor(img_a_padded, cv2.COLOR_BGR2RGB))
            
            # Apply normalization transforms only (images are already 512x512 from padding)
            # Create simple transform that just normalizes (no resize/crop since already correct size)
            normalize_A = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            
            A_tensor = normalize_A(img_a_pil).unsqueeze(0).to(opt.device)  # [1, 3, H, W] in range [-1, 1]
            
            # Run model inference
            model.real_A = A_tensor
            model.forward()  # This sets model.fake_B
            fake_B_tensor = model.fake_B  # [1, C, H, W]
            
            # Load and preprocess ground truth line drawing
            img_b = cv2.imread(line_drawing_path)
            if img_b is None:
                print(f"Skipping {filename}: Failed to read line drawing.")
                continue
            
            img_b_padded = resize_with_padding(img_b, size=opt.crop_size, interpolation=cv2.INTER_NEAREST)
            img_b_pil = PILImage.fromarray(cv2.cvtColor(img_b_padded, cv2.COLOR_BGR2RGB))
            if opt.output_nc == 1:
                img_b_pil = img_b_pil.convert("L")
            
            normalize_B = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,) if opt.output_nc == 1 else (0.5, 0.5, 0.5),
                                   (0.5,) if opt.output_nc == 1 else (0.5, 0.5, 0.5))
            ])
            B_tensor = normalize_B(img_b_pil).unsqueeze(0).to(opt.device)  # [1, C, H, W]
            
            # Convert to images for display
            real_A_img = tensor2im(A_tensor)
            fake_B_img = tensor2im(fake_B_tensor)
            real_B_img = tensor2im(B_tensor)
            
            # Concatenate images horizontally: original | result | target
            combined_img = np.concatenate([real_A_img, fake_B_img, real_B_img], axis=1)
            combined_pil = PILImage.fromarray(combined_img)
            combined_pil.save(os.path.join(results_dir, f"{name}_comparison{ext}"))
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(training_image_paths)} images...")
    
    print(f"\nCompleted! Processed {len(training_image_paths)} training images.")
    print(f"Results saved to: {results_dir}")
    print("="*50 + "\n")

    cleanup_ddp()
