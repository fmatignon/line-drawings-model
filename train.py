import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
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
    """Add LoRA adapters to UNet"""
    lora_attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = (
            None
            if name.endswith("attn1.processor")
            else unet.config.cross_attention_dim
        )
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        else:
            hidden_size = unet.config.block_out_channels[0]

        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=LORA_RANK,
        )

    unet.set_attn_processor(lora_attn_procs)
    return AttnProcsLayers(unet.attn_processors)


def main():
    # Setup accelerator for mixed precision
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device

    # Load dataset
    dataset = PairedImageDataset(
        "dataset/originals", "dataset/line_drawings", size=IMAGE_SIZE
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load pipeline
    print("Loading Stable Diffusion 1.5...")
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16
        if accelerator.mixed_precision == "fp16"
        else torch.float32,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    # Freeze VAE and text encoder
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)

    # Setup LoRA
    print("Setting up LoRA adapters...")
    lora_layers = setup_lora(pipe.unet)

    # Setup optimizer
    optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.L1Loss()

    # Prepare for training
    pipe.vae, pipe.unet, optimizer, dataloader = accelerator.prepare(
        pipe.vae, pipe.unet, optimizer, dataloader
    )

    # Training loop
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    pipe.unet.train()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, (originals, targets) in enumerate(dataloader):
            originals = originals.to(device)
            targets = targets.to(device)

            # Encode target images to latent space using VAE
            with torch.no_grad():
                # VAE expects input in [-1, 1] range, which we already have
                target_latents = pipe.vae.encode(targets).latent_dist.sample()
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
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.6f}")

        # Early stopping if loss plateaus (simple check)
        if epoch > 10 and avg_loss < 0.01:
            print(f"Loss converged, stopping early at epoch {epoch + 1}")
            break

    # Save LoRA weights
    print("Saving LoRA weights...")
    os.makedirs("lora_weights", exist_ok=True)
    unwrapped_unet = accelerator.unwrap_model(pipe.unet)
    unwrapped_unet.save_attn_procs("lora_weights")
    print("Training complete! LoRA weights saved to lora_weights/")


if __name__ == "__main__":
    main()
