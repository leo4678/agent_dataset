import argparse
import datasets
import random
import math
import pickle
import os
import json
import models

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from typing import Tuple

from lora import inject_lora
from mlx.utils import tree_flatten
from models import LoRALinear
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from stable_diffusion import StableDiffusion,StableDiffusionXL
from dataloader import DataLoader

import logging
import traceback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_dataset(sd, args):
    print("Loading datasets")
    dataset = load_dataset(
        "imagefolder",
        data_dir=args.data_dir
    )

    # 获取图像和文本描述
    column_names = dataset["train"].column_names
    image_column = column_names[0]
    caption_column = column_names[1]

    # 文本数据预处理流程
    def tokenize_captions(examples, is_train=True):
        captions = []
        #print(examples[caption_column])
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                captions.append(random.choice(caption)
                                if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        input_ids = sd.tokenizer.tokenize(captions)

        return input_ids

    # 图像数据预处理流程
    def resize_image(image, size, interpolation=Image.Resampling.BILINEAR):
        return image.resize(size, interpolation)

    def random_crop(image, size):
        width, height = image.size
        crop_width, crop_height = size
        if width < crop_width or height < crop_height:
            raise ValueError("Image size is smaller than the crop size")
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        right = left + crop_width
        bottom = top + crop_height
        return image.crop((left, top, right, bottom))

    def to_tensor(image):
        return np.array(image) / 255.0

    def normalize(tensor, mean, std):
        return (tensor - mean) / std

    def preprocess_image(image, resolution, mean=0.5, std=0.5):
        if not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image object")
        image = resize_image(image, (resolution, resolution))
        image = random_crop(image, (resolution, resolution))
        tensor = to_tensor(image)
        # print("zxdtest")
        # print(tensor.shape)
        tensor = normalize(tensor, mean, std)
        # print(tensor.shape)
        return tensor

    # 预处理
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        logging.info(f"Processing images: {images}")
        examples["pixel_values"] = [preprocess_image(image, args.resolution) for image in images]
        logging.info(f"Processed pixel values: {examples['pixel_values']}")
        examples["input_ids"] = tokenize_captions(examples)
        logging.info(f"Processed input IDs: {examples['input_ids']}")
        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    return train_dataloader

def setup_model(args):
    if args.model == "sd":
        sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=args.float16)
    else:
        sd = StableDiffusionXL("stabilityai/sdxl-turbo", float16=args.float16)
    sd.unet.freeze()
    sd.autoencoder.freeze()
    sd.text_encoder.freeze()

    # print(sd.unet)
    for name,layer in sd.unet.named_modules():
        name_cols=name.split('.')
        #print(name_cols)
        filter_names=['query_proj','key_proj','value_proj']
        if any(n in name_cols for n in filter_names) and isinstance(layer,nn.Linear):
            # print(name)
            inject_lora(sd.unet,name,layer)

    # print("=============================================================")
    # print(sd.unet)

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    return sd, optimizer

def loss(model, inputs, model_pred):
    logits, _ = model(inputs)
    logits = logits.astype(mx.float32)

    ce = nn.losses.cross_entropy(logits, model_pred)
    #loss = np.mean((model_pred - target) ** 2)
    return ce

def train_model(train_dataloader, sd, optimizer, args):
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.n_epochs * num_update_steps_per_epoch
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    print(f'num_train_epochs = {num_train_epochs}')

    global_step = 0
    first_epoch = 0
    initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
    )

    for epoch in range(first_epoch, num_train_epochs):
        sd.unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # loss_value_and_grad = nn.value_and_grad(sd.unet, loss)
            encoder_hidden_states = sd._get_text_conditioning("mountain", args.n_images)

            img = Image.open('dataset/mountain/mountain.jpg')

            W, H = (dim - dim % 64 for dim in (img.width, img.height))
            if W != img.width or H != img.height:
                print(f"Warning: image shape is not divisible by 64, downsampling to {W}x{H}")
                img = img.resize((W, H), Image.NEAREST)

            pixel_values = mx.array(np.array(img))
            img=(pixel_values[:, :, :3].astype(mx.float32) / 255) * 2 - 1

            # 图像编码成潜在向量
            start_step = sd.sampler.max_time * args.strength
            print("=========================")
            print(start_step)
            pixel_values = mx.array(np.array(batch["pixel_values"][step]))
            pixel_values = (pixel_values[:, :, :3].astype(mx.float32) / 255) * 2 - 1

            latents, _ = sd.autoencoder.encode(pixel_values[None])
            latents = mx.broadcast_to(latents, (args.n_images,) + latents.shape[1:])
            noisy_latents = sd.sampler.add_noise(latents, mx.array(start_step))
            print("=========================")
            print(noisy_latents)
            for x_t in tqdm(latents, total=int(args.steps * args.strength)):
                mx.eval(x_t)

            for t, t_prev in sd.sampler.timesteps(
                50, start_time=start_step, dtype=sd.dtype
            ):
                x_t_unet = mx.concatenate([noisy_latents] * 2, axis=0)
                t_unet = mx.broadcast_to(t, [len(x_t_unet)])
                eps_pred = sd.unet(x_t_unet, t_unet, encoder_x=encoder_hidden_states)
                eps_text, eps_neg = eps_pred.split(2)
                eps_pred = eps_neg + args.cfg * (eps_text - eps_neg)
                noisy_latents = sd.sampler.step(eps_pred, noisy_latents, t, t_prev)
                print(noisy_latents.shape)

            # (lvalue, _), grad = loss_value_and_grad(sd.unet, *batch)
            loss = np.mean(np.array(noisy_latents - eps_pred) ** 2)

            # 生成与潜在向量表示相同的高斯噪声
            # noise = mx.random.normal(latents.shape)
            # if args.noise_offset:
            #     noise += args.noise_offset * mx.random.normal((latents.shape[0], latents.shape[1], 1, 1))

            # bsz = latents.shape[0]
            # # 生成时间戳
            # timesteps = np.random.randint(0, sd.sampler.max_time, bsz)
            # timesteps = timesteps.astype(np.int64)

            # 添加噪声
            # noisy_latents = sd.sampler.add_noise(latents, noise, timesteps)
            # noisy_latents = sd.sampler.add_noise(latents, mx.array(start_step))

            # 获取文本输入
            # print(batch["input_ids"])
            # negative_text = ""
            # batch["input_ids"][step] += [sd.tokenizer.tokenize(negative_text)]
            # input_ids = mx.array(np.array(batch["input_ids"][step]))
            # print(input_ids.shape)
            # encoder_hidden_states = sd.text_encoder(input_ids)[0]
            # encoder_hidden_states = sd._get_text_conditioning("mountain", args.n_images)

            # target = noise
            # if sd.sampler.config.prediction_type == "epsilon":
            #     target = noise
            # elif sd.sampler.config.prediction_type == "v_prediction":
            #     target = sd.sampler.get_velocity(latents, noise, timesteps)
            # else:
            #     raise ValueError(
            #         f"Unknown prediction type {sd.sampler.config.prediction_type}")

            # x_T = sd.sampler.add_noise(latents, mx.array(start_step))
            # x_t_unet = mx.concatenate([x_T] * 2, axis=0)
            # t_unet = mx.broadcast_to(0, [len(x_t_unet)])
            # eps_pred = sd.unet(
            #     x_t_unet, t_unet, encoder_x=encoder_hidden_states, text_time=None
            # )

            # 预测噪声残差和均方误差损失
            #model_pred = sd.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # print("111111111111111111111111")
            # img = Image.open('dataset/mountain/mountain.jpg')
            #
            # W, H = (dim - dim % 64 for dim in (img.width, img.height))
            # if W != img.width or H != img.height:
            #     print(f"Warning: image shape is not divisible by 64, downsampling to {W}x{H}")
            #     img = img.resize((W, H), Image.NEAREST)
            #
            # pixel_values = mx.array(np.array(img))
            # img=(pixel_values[:, :, :3].astype(mx.float32) / 255) * 2 - 1
            #
            # latents = sd.generate_latents_from_image(
            #     img,
            #     "mountain",
            #     strength=args.strength,
            #     n_images=args.n_images,
            #     cfg_weight=args.cfg,
            #     num_steps=args.steps,
            #     negative_text=args.negative_prompt,
            #     seed=args.seed,
            # )
            # for x_t in tqdm(latents, total=int(args.steps * args.strength)):
            #     mx.eval(x_t)
            # print("111111111111111111111111")

            # optimizer.update(sd.unet, grad)
            # mx.eval(sd.unet.parameters(), optimizer.state, lvalue)
            #
            # if (step + 1) % args.gradient_accumulation_steps == 0:
            #     optimizer.step()
            #     optimizer.zero_grad()
            #
            # print("123456")
            # train_loss += lvalue.item() / args.gradient_accumulation_steps

            optimizer.init_single(mx.zeros())
            loss.backward()
            optimizer.step()
            last_loss=loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
                global_step += 1
            # if global_step % args.checkpointing_steps == 0:
            #     save_path = os.path.join(args.output, f"checkpoint-{global_step}")
            #     os.makedirs(save_path, exist_ok=True)
            #     torch.save(sd.unet.state_dict(), os.path.join(save_path, "unet.pth"))
            #     print(f"Saved state to {save_path}")

        if global_step >= max_train_steps:
            break
def save_lora_layers(sd, args):
    mx.savez(args.adapter_file, **dict(tree_flatten(sd.unet.trainable_parameters())))

def main():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument("--sd_specific_param", type=str, default="default_value", help="Specific param for SD finetuning")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--model", choices=["sd", "sdxl"], default="sdxl")
    parser.add_argument("--n_images", type=int, default=1)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--strength", type=float, default=0.9)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--n_rows", type=int, default=1)
    parser.add_argument("--decoding_batch_size", type=int, default=1)
    parser.add_argument("--no-float16", dest="float16", action="store_false")
    parser.add_argument("--quantize", "-q", action="store_true")
    parser.add_argument("--preload-models", action="store_true")
    parser.add_argument("--output", default="out.png")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--temp", type=float, default=0.8)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data_dir", type=str, default="dataset/mountain")
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4, help="Minibatch size.")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=400)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--noise_offset", type=float, default=0.1)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--add-eos-token", type=int, default=1)
    parser.add_argument("--adapter-file", type=str, default="adapters.npz")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    try:
        sd, optimizer = setup_model(args)
        train_dataloader = prepare_dataset(sd, args)

        train_model(train_dataloader, sd, optimizer, args)
        save_lora_layers(sd, args)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        logging.error("Traceback:")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    main()
