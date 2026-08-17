import torch
import torch.nn as nn
from model import NanoDiT
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision
from contextlib import nullcontext
from tqdm import tqdm
torch.set_float32_matmul_precision('high')

# --- Hyperparameters ---
NUM_CLASSES = 5
IMG_SIZE = 64
IMG_CHANNELS = 3
# DiT specific parameters
LATENT_DIM = 512
PATCH_SIZE = 2
MODEL_DEPTH = 12
MODEL_HEADS = 8

# Training parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
EPOCHS = 2000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16  # Automatic mixed precision dtype.
# Sampling parameters
SAMPLE_INTERVAL = 100  # Sample every N epochs
NUM_SAMPLES_PER_CLASS = 4  # Number of images to sample per class during evaluation

NUM_STEPS=50
# Others
CHECKPOINT_SAVE_INTERVAL = 100
main_folder = "/content/drive/MyDrive/diff"
DATA_DIR = f"{main_folder}/butterflies" # Directory where the dataset is stored.

class Diffusion(nn.Module):
    def __init__(self, beta_start, beta_end, timesteps):
        super().__init__()
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.register_buffer('beta_t', torch.linspace(self.beta_start, self.beta_end, self.timesteps))
        self.register_buffer('alpha_t',1- self.beta_t)
        self.register_buffer('alphabar_t', torch.cumprod(self.alpha_t, dim=0))

        self.eps_theta = NanoDiT(
                input_size=IMG_SIZE,
                patch_size=PATCH_SIZE,
                in_channels=IMG_CHANNELS,
                hidden_size=LATENT_DIM,
                depth=MODEL_DEPTH,
                num_heads=MODEL_HEADS,
                num_classes=NUM_CLASSES,
                timestep_freq_scale=1000,
            )

    def forward_process(self, x_0, eps, t):
        alphabar_t = self.alphabar_t[t].view(-1, 1, 1, 1)
        x_t = torch.sqrt(alphabar_t)*x_0 + torch.sqrt(1 - alphabar_t)*eps
        return x_t

    @torch.no_grad()
    def sample(self, target_classes_list, num_steps,num_samples_per_cls=1):
        """Generate images for specified target classes."""
        self.eps_theta.eval()
        num_target_cls = len(target_classes_list)
        total_images_to_sample = num_samples_per_cls * num_target_cls

        # Initial state
        x_t = torch.randn((total_images_to_sample, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), device=DEVICE)

        # Prepare conditional labels
        sample_cls_labels_list = []
        for c_idx in target_classes_list:
            sample_cls_labels_list.extend([c_idx] * num_samples_per_cls)
        conditional_labels = torch.tensor(sample_cls_labels_list, device=DEVICE).long()

        y = conditional_labels

        steps = torch.linspace(self.timesteps-1, 0, num_steps, dtype=torch.int, device=DEVICE)

        for i in tqdm(range(num_steps)):
            t = steps[i].unsqueeze(0)
            alphabar_t = self.alphabar_t[t].view(-1, 1, 1, 1)
            pred_eps = self.eps_theta(x_t, t/self.timesteps, y)
            x_hat = (x_t - (1-alphabar_t).sqrt()*pred_eps)/alphabar_t.sqrt()
            if i == len(steps)-1:
                x_t = x_hat
                break
            prev_t = steps[i+1]
            alphabar_prevt = self.alphabar_t[prev_t].view(-1, 1, 1, 1)
            x_prevt = alphabar_prevt.sqrt()*x_hat + (1-alphabar_prevt).sqrt()*pred_eps
            x_t = x_prevt

        images = (x_t + 1) / 2.0  # De-normalize from [-1, 1] to [0, 1]
        images = torch.clamp(images, 0.0, 1.0)

        self.eps_theta.train() # Set model to train.
        return images, conditional_labels

        
    
    def loss(self, x_0, y):
        t = torch.randint(0, self.timesteps, (x_0.shape[0],)).to(DEVICE)
        eps = torch.randn_like(x_0)
        x_t = self.forward_process(x_0, eps, t)
        pred_eps = self.eps_theta(x_t, t/self.timesteps, y)
        return torch.nn.functional.mse_loss(eps, pred_eps)
    

diffusion = Diffusion(beta_start=0.0001, beta_end=0.02, timesteps=1000).to(DEVICE)

optimizer = torch.optim.Adam(diffusion.eps_theta.parameters(), lr=LEARNING_RATE)
scaler = torch.GradScaler() if AMP_DTYPE is not None else None
amp_context = (
    torch.autocast(device_type=torch.device(DEVICE).type, dtype=AMP_DTYPE) 
    if AMP_DTYPE is not None
    else nullcontext()
)
if AMP_DTYPE:
    print(f"Using automatic mixed-precision in {AMP_DTYPE} (change if needed).")

# --- Dataset and DataLoader  ---
ds_trfs = transforms.Compose(
    [
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ]
)
train_dataset = torchvision.datasets.ImageFolder(DATA_DIR, transform=ds_trfs)
train_classes = list(set(train_dataset.class_to_idx.values()))
assert NUM_CLASSES == len(train_classes)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,  # Adjust based on your system
    pin_memory=True,  # Useful when training on GPU
    drop_last=True,
    prefetch_factor=2, # Adjust based on your system
)

# --- Training Loop ---
print(f"Training on {DEVICE}")
print(f"Using custom model: {type(diffusion.eps_theta).__name__}")
print(f"Model Parameters: {sum(p.numel() for p in diffusion.eps_theta.parameters() if p.requires_grad)}")



for epoch in range(EPOCHS):
    diffusion.eps_theta.train()
    
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

    for step, (real_images, class_ids) in enumerate(progress_bar):
        optimizer.zero_grad()

        real_images = real_images.to(DEVICE, non_blocking=True)
        class_ids = class_ids.to(DEVICE, non_blocking=True)

        with amp_context:
            loss = diffusion.loss(real_images, class_ids)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update tqdm progress bar with loss
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    # --- Perform Sampling and Save Images (Intermediate Evaluation) ---
    if (epoch + 1) % SAMPLE_INTERVAL == 0 or epoch == EPOCHS - 1:
        print(f"\nSampling images at epoch {epoch + 1}...")
        classes_to_sample_list = list(range(min(NUM_CLASSES, 5)))
        generated_sample_images, _ = diffusion.sample(classes_to_sample_list, NUM_STEPS, num_samples_per_cls=NUM_SAMPLES_PER_CLASS)
        # Save as a grid
        if generated_sample_images.nelement() > 0:  # Check if any images were generated
            grid = torchvision.utils.make_grid(generated_sample_images, nrow=NUM_SAMPLES_PER_CLASS)
            torchvision.utils.save_image(grid, f"{main_folder}/ddim/sample_epoch_{epoch + 1}.png")
            print(f"Saved sample images to sample_epoch_{epoch + 1}.png")
        print("-" * 30)
        
        # Optional: Save model checkpoint
        if (epoch + 1) % CHECKPOINT_SAVE_INTERVAL == 0:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": diffusion.eps_theta.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            }
    
            torch.save(checkpoint, f"{main_folder}/ddim/checkpoint_epoch_{epoch + 1}.pth")
        
print("Training finished.")
