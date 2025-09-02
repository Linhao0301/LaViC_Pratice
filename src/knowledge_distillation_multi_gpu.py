#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LaViC Code: Visual Knowledge Self-Distillation (Multi-GPU V2)
-------------------------------------------------------
Enhanced multi-GPU version with:
1. Pre-training GPU memory check
2. Optimized disk space usage (avoid system partition)
3. LoRA-only checkpoint saving
4. Better error handling and monitoring
"""

import argparse
import json
import math
import os
import subprocess
import sys

import pytorch_lightning as pl
import torch
from PIL import Image, UnidentifiedImageError
from peft import LoraConfig, get_peft_model, TaskType
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

# 设置环境变量
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

# 设置临时目录到有足够空间的分区，避免checkpoint保存时空间不足
TMPDIR = "/root/autodl-tmp/tmp"
os.makedirs(TMPDIR, exist_ok=True)
os.environ["TMPDIR"] = TMPDIR
os.environ["TEMP"] = TMPDIR
os.environ["TMP"] = TMPDIR

print(f"🗂️  [INIT] Using temp directory: {TMPDIR}")


def check_gpu_memory():
    """
    检查GPU显存使用情况，如果显存被大量占用则报错
    """
    print("🔍 [GPU Check] Checking GPU memory usage...")
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, check=True)
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    idx, name, used, total = parts[0], parts[1], int(parts[2]), int(parts[3])
                    usage_pct = (used / total) * 100
                    gpu_info.append({
                        'index': idx,
                        'name': name, 
                        'used': used,
                        'total': total,
                        'usage_pct': usage_pct
                    })
        
        print(f"📊 [GPU Check] Found {len(gpu_info)} GPUs:")
        
        problematic_gpus = []
        for gpu in gpu_info:
            status = "🟢" if gpu['usage_pct'] < 10 else "🟡" if gpu['usage_pct'] < 50 else "🔴"
            print(f"   GPU {gpu['index']}: {gpu['name']}")
            print(f"   {status} Memory: {gpu['used']}MB / {gpu['total']}MB ({gpu['usage_pct']:.1f}%)")
            
            # 如果显存使用超过50%，标记为有问题
            if gpu['usage_pct'] > 50:
                problematic_gpus.append(gpu)
        
        if problematic_gpus:
            print("\n❌ [GPU Check] ERROR: High GPU memory usage detected!")
            print("🔧 [GPU Check] Please check and kill existing processes:")
            print("   - Run: nvidia-smi")
            print("   - Kill processes: kill -9 <PID>")
            print("   - Or restart: sudo systemctl restart nvidia-persistenced")
            
            for gpu in problematic_gpus:
                print(f"   GPU {gpu['index']}: {gpu['usage_pct']:.1f}% used ({gpu['used']}MB/{gpu['total']}MB)")
            
            return False
        
        print("✅ [GPU Check] All GPUs have sufficient free memory")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  [GPU Check] Warning: Could not check GPU memory: {e}")
        return True  # 继续执行，但给出警告
    except Exception as e:
        print(f"⚠️  [GPU Check] Warning: GPU check failed: {e}")
        return True


def check_disk_space(min_free_gb=5):
    """
    检查磁盘空间，确保有足够空间用于训练
    """
    print("💾 [Disk Check] Checking disk space...")
    
    try:
        # 检查当前工作目录的磁盘空间
        statvfs = os.statvfs('.')
        free_bytes = statvfs.f_frsize * statvfs.f_bavail
        free_gb = free_bytes / (1024**3)
        
        print(f"📁 [Disk Check] Available space: {free_gb:.2f} GB")
        
        if free_gb < min_free_gb:
            print(f"❌ [Disk Check] ERROR: Insufficient disk space!")
            print(f"   Required: {min_free_gb} GB, Available: {free_gb:.2f} GB")
            return False
        
        print("✅ [Disk Check] Sufficient disk space available")
        return True
        
    except Exception as e:
        print(f"⚠️  [Disk Check] Warning: Could not check disk space: {e}")
        return True


# ---------------------------------------------------------
# Special tokens and template (same as original)
# ---------------------------------------------------------
IMAGE_TOKENS = [
    "<ItemImageEmb1>", "<ItemImageEmb2>", "<ItemImageEmb3>",
    "<ItemImageEmb4>", "<ItemImageEmb5>"
]

PROMPT_TEMPLATE = (
    "You are a helpful assistant.\n"
    "Given an Amazon product's title and its image, please provide a detailed, visually grounded description of the product "
    "that would help someone decide whether to purchase it. "
    "Focus on the product's appearance, features, and any other visually informative aspects. "
    "Do not mention the product's title in your answer. "
    "This product's title is: {title}\n"
    f"{''.join(IMAGE_TOKENS)}\n\n"
    "Assistant:"
)


# ---------------------------------------------------------
# Dataset (same as original but with better error handling)
# ---------------------------------------------------------
class ImageDescriptionDataset(Dataset):
    def __init__(self, data_source, images_dir, is_training=True, default_image_size=(336, 336), max_samples=None):
        super().__init__()
        self.images_dir = images_dir
        self.is_training = is_training
        self.default_image = Image.new('RGB', default_image_size, (255, 255, 255))
        self.max_samples = max_samples

        self.data = []
        if data_source.endswith('.json'):
            with open(data_source, 'r', encoding='utf-8') as f:
                data_json = json.load(f)
            for asin, item_data in data_json.items():
                title = item_data.get("title", "No Title")
                image_descs = item_data.get("image_descriptions_llava_cleaned", {})
                for image_name, desc in image_descs.items():
                    image_path = os.path.join(images_dir, image_name)
                    if os.path.exists(image_path):
                        self.data.append({
                            "title": title,
                            "image_path": image_path,
                            "description": desc
                        })
        elif data_source.endswith('.jsonl'):
            with open(data_source, 'r', encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    title = entry.get("title", "No Title")
                    image_name = entry.get("image_name", "")
                    desc = entry.get("image_description_llava_cleaned", "")
                    image_path = os.path.join(images_dir, image_name)
                    if os.path.exists(image_path):
                        self.data.append({
                            "title": title,
                            "image_path": image_path,
                            "description": desc
                        })
        else:
            raise ValueError("Data source must be either .json or .jsonl")
        
        if self.max_samples is not None and len(self.data) > self.max_samples:
            print(f"[INFO] Limiting dataset from {len(self.data)} to {self.max_samples} samples")
            self.data = self.data[:self.max_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image_path"]
        
        image = self.default_image
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                if image.size[0] == 0 or image.size[1] == 0:
                    image = self.default_image
            except (UnidentifiedImageError, OSError, IOError, Exception):
                image = self.default_image

        return {
            "title": item["title"],
            "image": image,
            "description": item["description"]
        }


# ---------------------------------------------------------
# Data Collator (same as original)
# ---------------------------------------------------------
class DataCollator:
    def __init__(self, processor, tokenizer, max_length, prompt_template):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = prompt_template
        self.image_token_ids = [
            self.tokenizer.convert_tokens_to_ids(tk) for tk in IMAGE_TOKENS
        ]

    def __call__(self, batch):
        prompts = []
        target_texts = []
        images = []

        for item in batch:
            try:
                title = item.get("title", "No Title")
                desc = item.get("description", "No Description")
                
                if not title.strip():
                    title = "No Title"
                if not desc.strip():
                    desc = "No Description"
                    
                prompt = self.prompt_template.format(title=title)
                prompts.append(prompt)
                target_texts.append(desc)
                images.append(item["image"])
            except Exception:
                continue

        if not prompts:
            prompts = ["No Title"]
            target_texts = ["No Description"] 
            images = [Image.new('RGB', (336, 336), color=(255, 255, 255))]

        full_prompts = [p + t for p, t in zip(prompts, target_texts)]

        tokenized_prompts = self.tokenizer(
            prompts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )
        tokenized_full_prompts = self.tokenizer(
            full_prompts,
            max_length=self.max_length,
            padding='longest',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = tokenized_full_prompts['input_ids']
        attention_mask = tokenized_full_prompts['attention_mask']
        labels = input_ids.clone()

        prompt_lengths = [len(x) for x in tokenized_prompts['input_ids']]
        for i in range(len(labels)):
            labels[i, :prompt_lengths[i]] = -100

        labels[labels == self.tokenizer.pad_token_id] = -100

        image_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for b_idx in range(input_ids.size(0)):
            for tk_id in self.image_token_ids:
                positions = (input_ids[b_idx] == tk_id).nonzero(as_tuple=True)
                image_token_mask[b_idx, positions] = True

        images_processed = self.processor.image_processor(images, return_tensors='pt')
        images_tensor = images_processed['pixel_values']

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'images': images_tensor,
            'image_token_mask': image_token_mask
        }


# ---------------------------------------------------------
# Enhanced Lightning Module for Multi-GPU
# ---------------------------------------------------------
class PretrainVisionModelMultiGPU(pl.LightningModule):
    def __init__(self, model, processor, tokenizer, args):
        super().__init__()
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        self.args = args

        self.data_collator = DataCollator(
            processor,
            tokenizer,
            max_length=args.max_length,
            prompt_template=PROMPT_TEMPLATE
        )
        # 只保存必要的超参数，避免保存大模型
        self.save_hyperparameters(ignore=['model', 'processor', 'tokenizer'])

        self.val_loss_sum = 0.0
        self.val_token_count = 0

    def forward(self, input_ids, attention_mask, images, image_token_mask, labels=None):
        device = next(self.model.parameters()).device
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        image_token_mask = image_token_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids)

        if images is not None:
            images = images.to(device, dtype=torch.float16)
            B, num_views, C, H, W = images.shape

            images_reshaped = images.view(B * num_views, C, H, W)
            vision_outputs = self.model.vision_tower(images_reshaped)

            cls_states = vision_outputs.last_hidden_state[:, 0, :].view(B, num_views, -1)
            cls_states = self.model.multi_modal_projector(cls_states)

            for b_idx in range(B):
                positions = torch.nonzero(image_token_mask[b_idx], as_tuple=False).squeeze(-1)
                pos_count = min(len(positions), num_views)
                for i in range(pos_count):
                    col = positions[i].item()
                    inputs_embeds[b_idx, col, :] = cls_states[b_idx, i, :]

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def training_step(self, batch, batch_idx):
        inputs = self.data_collator(batch)
        outputs = self(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            images=inputs['images'],
            image_token_mask=inputs['image_token_mask'],
            labels=inputs['labels']
        )
        loss = outputs.loss
        
        effective_batch_size = len(batch) * self.trainer.world_size if self.trainer.world_size > 1 else len(batch)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=effective_batch_size, sync_dist=True)
        
        # 记录GPU利用率（仅在主进程）
        if self.trainer.is_global_zero and batch_idx % 20 == 0:
            try:
                gpu_memory = torch.cuda.max_memory_allocated() / 1024**3
                self.log('gpu_memory_gb', gpu_memory, on_step=True, prog_bar=False)
            except:
                pass
                
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = self.data_collator(batch)
        outputs = self(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            images=inputs['images'],
            image_token_mask=inputs['image_token_mask'],
            labels=inputs['labels']
        )
        val_loss = outputs.loss
        num_tokens = (inputs['labels'] != -100).sum().item()

        self.val_loss_sum += val_loss.item() * num_tokens
        self.val_token_count += num_tokens
        return val_loss

    def on_validation_epoch_end(self):
        # 在多GPU环境中同步验证结果
        if self.trainer.world_size > 1:
            val_loss_sum_tensor = torch.tensor(self.val_loss_sum, device=self.device)
            val_token_count_tensor = torch.tensor(self.val_token_count, device=self.device)
            
            torch.distributed.all_reduce(val_loss_sum_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(val_token_count_tensor, op=torch.distributed.ReduceOp.SUM)
            
            total_val_loss_sum = val_loss_sum_tensor.item()
            total_val_token_count = val_token_count_tensor.item()
        else:
            total_val_loss_sum = self.val_loss_sum
            total_val_token_count = self.val_token_count

        if total_val_token_count > 0:
            avg_val_loss = total_val_loss_sum / total_val_token_count
        else:
            avg_val_loss = float('inf')
        ppl = math.exp(avg_val_loss) if avg_val_loss < 20 else float('inf')

        self.log('val_loss', avg_val_loss, prog_bar=True, sync_dist=True)
        self.log('val_perplexity', ppl, prog_bar=True, sync_dist=True)

        # Save metrics (仅在主进程)
        if self.trainer.is_global_zero:
            with open(os.path.join(self.args.output_dir, f"val_metrics_epoch_{self.current_epoch+1}.txt"), "w") as f:
                f.write(f"Val Loss: {avg_val_loss}\nVal PPL: {ppl}\n")

        self.val_loss_sum = 0.0
        self.val_token_count = 0

    def configure_optimizers(self):
        params_to_optimize = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params_to_optimize,
            lr=self.args.lr,
            weight_decay=self.args.weight_decay
        )
        return optimizer


# ---------------------------------------------------------
# Enhanced LoRA-only Checkpoint Callback
# ---------------------------------------------------------
class LoRAOnlyCheckpoint(ModelCheckpoint):
    """
    自定义checkpoint回调，只保存LoRA参数，避免保存完整模型
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _save_checkpoint(self, trainer, filepath):
        # 只保存LoRA adapter，不保存完整模型
        if trainer.is_global_zero:
            print(f"💾 [Checkpoint] Saving LoRA adapter to {filepath}")
            
            # 创建LoRA专用目录
            lora_dir = filepath.replace('.ckpt', '_lora_adapter')
            os.makedirs(lora_dir, exist_ok=True)
            
            # 保存LoRA参数
            trainer.lightning_module.model.save_pretrained(lora_dir)
            
            # 保存训练状态（用于恢复训练，但不包含完整模型）
            checkpoint = {
                'epoch': trainer.current_epoch,
                'global_step': trainer.global_step,
                'pytorch-lightning_version': pl.__version__,
                'state_dict': {k: v for k, v in trainer.lightning_module.state_dict().items() 
                              if 'lora' in k.lower() or 'adapter' in k.lower()},  # 只保存LoRA相关参数
                'optimizer_states': [opt.state_dict() for opt in trainer.optimizers],
                'lr_schedulers': [],
                'hyper_parameters': trainer.lightning_module.hparams,
                'val_loss': trainer.callback_metrics.get('val_loss', float('inf'))
            }
            
            # 保存轻量级checkpoint
            torch.save(checkpoint, filepath)
            print(f"✅ [Checkpoint] LoRA adapter saved: {lora_dir}")


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        all_params += num_params
        if param.requires_grad:
            trainable_params += num_params

    pct = 100 * trainable_params / all_params if all_params > 0 else 0
    print(f"📊 [Model] Trainable: {trainable_params:,} | Total: {all_params:,} | Ratio: {pct:.2f}%")


def find_vision_linear_layer_names(vision_model, prefix="vision_tower"):
    import torch
    linear_names = []
    for name, module in vision_model.named_modules():
        if isinstance(module, torch.nn.Linear):
            full_name = f"{prefix}.{name}" if prefix else name
            linear_names.append(full_name)
    return linear_names


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU Vision Distillation with Enhanced Features")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-v1.6-mistral-7b-hf")
    parser.add_argument("--train_data", type=str, default="../data/item2meta_train.json")
    parser.add_argument("--val_data", type=str, default="../data/item2meta_valid.jsonl")
    parser.add_argument("--train_images_dir", type=str, default="../data/train_images")
    parser.add_argument("--val_images_dir", type=str, default="../data/valid_images")
    parser.add_argument("--output_dir", type=str, default="./out_distilled_multi_gpu")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    
    # Multi-GPU specific
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--strategy", type=str, default="ddp")
    parser.add_argument("--skip_gpu_check", action="store_true", help="Skip GPU memory check")
    
    args = parser.parse_args()

    print("🚀 LaViC Multi-GPU Training V2 Starting...")
    print(f"📊 Configuration: {args.devices} GPUs, batch_size={args.batch_size}, strategy={args.strategy}")

    # 1) 系统检查
    if not args.skip_gpu_check:
        if not check_gpu_memory():
            print("❌ GPU memory check failed. Exiting...")
            sys.exit(1)
    
    if not check_disk_space():
        print("❌ Disk space check failed. Exiting...")
        sys.exit(1)

    # 2) 初始化
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # 3) 模型加载（优化显存使用）
    print("📥 [Model] Loading base model...")
    torch.cuda.empty_cache()  # 清理显存
    
    base_model = LlavaForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=None,  # 让Lightning管理设备
    )

    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer

    # 4) 添加特殊token
    special_tokens_dict = {'additional_special_tokens': IMAGE_TOKENS}
    tokenizer.add_special_tokens(special_tokens_dict)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    base_model.resize_token_embeddings(len(tokenizer))

    # 5) LoRA配置
    vision_linear_names = find_vision_linear_layer_names(base_model.vision_tower, prefix="vision_tower")
    projector_linear_names = find_vision_linear_layer_names(base_model.multi_modal_projector, prefix="multi_modal_projector")
    target_modules = vision_linear_names + projector_linear_names

    print(f"🎯 [LoRA] Applying LoRA to {len(target_modules)} linear layers")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules
    )
    lora_model = get_peft_model(base_model, lora_config)
    print_trainable_parameters(lora_model)

    # 6) Lightning模块
    pl_model = PretrainVisionModelMultiGPU(lora_model, processor, tokenizer, args)

    # 7) 数据加载
    train_dataset = ImageDescriptionDataset(args.train_data, args.train_images_dir, is_training=True, max_samples=args.max_samples)
    val_dataset = ImageDescriptionDataset(args.val_data, args.val_images_dir, is_training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    print(f"📈 [Data] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 8) 训练器配置
    checkpoint_callback = LoRAOnlyCheckpoint(
        dirpath=args.output_dir,
        filename='multi_gpu_epoch{epoch}-val_loss{val_loss:.4f}',
        save_top_k=1,
        monitor='val_loss',
        mode='min',
    )

    strategy = DDPStrategy(
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        accelerator="gpu",
        devices=args.devices,
        strategy=strategy,
        callbacks=[checkpoint_callback],
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        # 优化内存使用
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    print(f"\n🚀 [Training] Starting multi-GPU training...")
    print(f"🔧 [Config] {args.devices} GPUs, {args.strategy} strategy, 16-mixed precision")
    
    # 9) 开始训练
    trainer.fit(pl_model, train_loader, val_loader)

    # 10) 保存最终LoRA adapter（仅主进程）
    if trainer.is_global_zero:
        best_lora_dir = os.path.join(args.output_dir, "vision_lora_adapter_best_multi_gpu")
        pl_model.model.save_pretrained(best_lora_dir)
        print(f"🎉 [Success] Best LoRA adapter saved: {best_lora_dir}")
        
        # 显示训练总结
        print("\n📊 [Summary] Multi-GPU Training Complete!")
        print(f"   • Output directory: {args.output_dir}")
        print(f"   • LoRA adapter: {best_lora_dir}")
        print(f"   • Trainable parameters: {sum(p.numel() for p in lora_model.parameters() if p.requires_grad):,}")


if __name__ == "__main__":
    main()
