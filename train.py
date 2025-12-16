import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from peft import LoraConfig
from accelerate import Accelerator

# Hardcoded hyperparameters
IMAGE_SIZE = 512
BATCH_SIZE = 1
LORA_RANK = 16
LORA_ALPHA = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
MODEL_ID = "runwayml/stable-diffusion-v1-5"


class PairedImageDataset(Dataset):
    def __init__(self, originals_dir, targets_dir, size=512):
        self.size = size
        self.originals_dir = Path(originals_dir)
        self.targets_dir = Path(targets_dir)

        # Find all matching pairs
        originals = set(f.name for f in self.originals_dir.glob("*.png"))
        targets = set(f.name for f in self.targets_dir.glob("*.png"))
        self.pairs = sorted(list(originals & targets))

        print(f"Found {len(self.pairs)} paired images")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        filename = self.pairs[idx]
        original_path = self.originals_dir / filename
        target_path = self.targets_dir / filename

        # Load images
        original = Image.open(original_path).convert("RGB")
        target = Image.open(target_path).convert("RGB")

        # Resize maintaining aspect ratio with padding
        original = self._resize_with_padding(original)
        target = self._resize_with_padding(target)

        # Random horizontal flip for augmentation
        if random.random() > 0.5:
            original = original.transpose(Image.FLIP_LEFT_RIGHT)
            target = target.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to tensor and normalize to [-1, 1]
        original_tensor = self._to_tensor(original)
        target_tensor = self._to_tensor(target)

        return original_tensor, target_tensor

    def _resize_with_padding(self, img):
        """Resize image to self.size x self.size maintaining aspect ratio"""
        img.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        new_img = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        paste_x = (self.size - img.width) // 2
        paste_y = (self.size - img.height) // 2
        new_img.paste(img, (paste_x, paste_y))
        return new_img

    def _to_tensor(self, img):
        """Convert PIL image to tensor normalized to [-1, 1]"""
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # Normalize to [-1, 1]
        arr = arr.transpose(2, 0, 1)  # HWC to CHW
        return torch.from_numpy(arr)


def setup_lora(unet):
    """Add LoRA adapters to UNet using PEFT"""
    # Configure LoRA using PEFT (modern diffusers approach)
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    # Add LoRA adapter to UNet
    unet.add_adapter(lora_config)

    return unet


def main():
    # Explicitly check for CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"PyTorch version: {torch.__version__}")
        print(
            f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )
        # Use fp16 mixed precision for CUDA (saves memory and speeds up training)
        mixed_precision = "fp16"
    else:
        print("Warning: CUDA not available, using CPU (training will be very slow)")
        print("Make sure you have CUDA-enabled PyTorch installed for Windows")
        mixed_precision = "no"

    # Setup accelerator for mixed precision
    accelerator = Accelerator(mixed_precision=mixed_precision)
    device = accelerator.device
    print(f"Training device: {device}")

    # Load dataset and split into train/validation
    full_dataset = PairedImageDataset(
        "dataset/originals", "dataset/line_drawings", size=IMAGE_SIZE
    )

    # Split: use 10 images for validation, rest for training
    total_size = len(full_dataset)
    val_size = min(10, total_size)
    train_size = total_size - val_size

    # Create train and validation datasets
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_size))

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    print(
        f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}"
    )

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load pipeline
    print("Loading Stable Diffusion 1.5...")
    # Load in float32 first to ensure LoRA adapters are created in fp32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # Load in float32 first
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Freeze VAE and text encoder
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Setup LoRA in fp32 first
    print("Setting up LoRA adapters...")
    setup_lora(pipe.unet)

    # For mixed precision: keep LoRA parameters in fp32 for stable gradients
    # The base UNet can be in fp16, but LoRA params stay fp32
    if accelerator.mixed_precision == "fp16":
        print("Converting base model to fp16 (keeping LoRA in fp32)...")
        # Convert base UNet to fp16, but explicitly keep LoRA params in fp32
        for name, param in pipe.unet.named_parameters():
            if "lora" not in name.lower():
                # Only convert non-LoRA parameters to fp16
                param.data = param.data.to(dtype=torch.float16)
        # VAE can be in fp16 for encoding/decoding
        pipe.vae = pipe.vae.to(dtype=torch.float16)

    # Setup optimizer - only train LoRA parameters
    # Filter to get only LoRA parameters (those that require grad)
    lora_layers = filter(lambda p: p.requires_grad, pipe.unet.parameters())
    trainable_params = list(lora_layers)
    print(f"Training {sum(p.numel() for p in trainable_params)} LoRA parameters")
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    criterion = torch.nn.L1Loss()

    # Prepare for training
    pipe.vae, pipe.unet, optimizer, train_dataloader, val_dataloader = (
        accelerator.prepare(
            pipe.vae, pipe.unet, optimizer, train_dataloader, val_dataloader
        )
    )

    # Training loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    pipe.unet.train()

    for epoch in range(NUM_EPOCHS):
        # Training phase
        pipe.unet.train()
        train_loss = 0.0
        num_train_batches = 0

        for batch_idx, (originals, targets) in enumerate(train_dataloader):
            originals = originals.to(device)
            targets = targets.to(device)

            # Encode target images to latent space using VAE
            with torch.no_grad():
                # VAE expects input in [-1, 1] range, which we already have
                # Convert to VAE's dtype (fp16 if using mixed precision)
                vae_dtype = next(pipe.vae.parameters()).dtype
                targets_encoded = targets.to(dtype=vae_dtype)
                target_latents = pipe.vae.encode(targets_encoded).latent_dist.sample()
                target_latents = target_latents * pipe.vae.config.scaling_factor

            # Standard SD training: add noise to target and predict noise
            noise = torch.randn_like(target_latents)
            timesteps = torch.randint(
                0,
                pipe.scheduler.config.num_train_timesteps,
                (target_latents.shape[0],),
                device=device,
            )
            noisy_latents = pipe.scheduler.add_noise(target_latents, noise, timesteps)

            # Dummy prompt for text conditioning (required by SD architecture)
            prompt = ""
            tokenizer = pipe.tokenizer
            text_encoder = pipe.text_encoder
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

            # Forward pass through UNet
            model_pred = pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=text_embeddings,
            ).sample

            # Predict noise (SD training objective)
            if pipe.scheduler.config.prediction_type == "epsilon":
                target = noise
            elif pipe.scheduler.config.prediction_type == "v_prediction":
                target = pipe.scheduler.get_velocity(target_latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {pipe.scheduler.config.prediction_type}"
                )

            loss = criterion(model_pred, target)

            # Backward pass
            accelerator.backward(loss)

            # Clip gradients to prevent explosion
            # Use torch's clip_grad_norm_ directly to avoid Accelerate's unscaling issues
            if accelerator.sync_gradients:
                # Get only trainable parameters that have gradients
                params_with_grad = [p for p in trainable_params if p.grad is not None]
                if params_with_grad:
                    # Use torch's clip_grad_norm_ directly (works with fp32 gradients)
                    torch.nn.utils.clip_grad_norm_(params_with_grad, max_norm=1.0)

            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = (
            train_loss / num_train_batches if num_train_batches > 0 else 0.0
        )

        # Validation phase
        pipe.unet.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for originals, targets in val_dataloader:
                originals = originals.to(device)
                targets = targets.to(device)

                # Encode target images to latent space using VAE
                # Convert to VAE's dtype (fp16 if using mixed precision)
                vae_dtype = next(pipe.vae.parameters()).dtype
                targets_encoded = targets.to(dtype=vae_dtype)
                target_latents = pipe.vae.encode(targets_encoded).latent_dist.sample()
                target_latents = target_latents * pipe.vae.config.scaling_factor

                # Standard SD training: add noise to target and predict noise
                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0,
                    pipe.scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],),
                    device=device,
                )
                noisy_latents = pipe.scheduler.add_noise(
                    target_latents, noise, timesteps
                )

                # Dummy prompt for text conditioning
                prompt = ""
                tokenizer = pipe.tokenizer
                text_encoder = pipe.text_encoder
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]

                # Forward pass through UNet
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=text_embeddings,
                ).sample

                # Predict noise (SD training objective)
                if pipe.scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif pipe.scheduler.config.prediction_type == "v_prediction":
                    target = pipe.scheduler.get_velocity(
                        target_latents, noise, timesteps
                    )
                else:
                    raise ValueError(
                        f"Unknown prediction type {pipe.scheduler.config.prediction_type}"
                    )

                loss = criterion(model_pred, target)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        # Early stopping if loss plateaus (simple check)
        if epoch > 10 and avg_train_loss < 0.01:
            print(f"Loss converged, stopping early at epoch {epoch + 1}")
            break

    # Save LoRA weights
    print("Saving LoRA weights...")
    os.makedirs("lora_weights", exist_ok=True)
    unwrapped_unet = accelerator.unwrap_model(pipe.unet)
    # Save using PEFT's save_pretrained method
    unwrapped_unet.save_pretrained("lora_weights")
    print("Training complete! LoRA weights saved to lora_weights/")


if __name__ == "__main__":
    main()
