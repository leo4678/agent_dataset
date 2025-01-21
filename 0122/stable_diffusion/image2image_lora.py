import argparse
import random
import math
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from lora import inject_lora
from mlx.utils import tree_flatten
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from stable_diffusion import StableDiffusion
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
        tensor = normalize(tensor, mean, std)
        return tensor

    # 预处理
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [preprocess_image(image, args.resolution) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    train_dataset = dataset["train"].with_transform(preprocess_train)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    print(train_dataloader.dataset)

    return train_dataloader

def setup_model(args):
    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=args.float16)
    sd.unet.freeze()
    sd.autoencoder.freeze()
    sd.text_encoder.freeze()

    for name,layer in sd.unet.named_modules():
        name_cols=name.split('.')
        filter_names=['query_proj','key_proj','value_proj']
        if any(n in name_cols for n in filter_names) and isinstance(layer,nn.Linear):
            inject_lora(sd.unet,name,layer)

    optimizer = optim.Adam(learning_rate=args.learning_rate)
    return sd, optimizer


def loss(pred, targets):
    logits = pred
    logits = logits.astype(mx.float32)

    ce = nn.losses.mse_loss(logits, targets)
    return ce.mean()


# 梯度下降算法可以分为下面4个步骤：
# 1、正向传播计算损失值
# 2、反向传播计算梯度
# 3、利用梯度更新参数
# 4、重复1、2、3的步骤，直到获取较小的损失
def train_model(train_dataloader, sd, optimizer, args):
    print(f'len(train_dataloader) = {len(train_dataloader)}')
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.n_epochs * num_update_steps_per_epoch
    print(f'max_train_steps = {max_train_steps}')
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    print(f'num_train_epochs = {num_train_epochs}')

    global_step = 0
    first_epoch = 0
    initial_global_step = 0
    losses = []
    start = time.perf_counter()

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
    )

    for epoch in range(first_epoch, num_train_epochs):
        sd.unet.train()
        # for step, batch in enumerate(train_dataloader):
        #     print(f"Step: {step}")
        #     print(batch["pixel_values"])
        #     if step >= 5:  # 打印前 5 个批次后退出
        #         break

        for step, batch in enumerate(train_dataloader):
            loss_value_and_grad = nn.value_and_grad(sd.unet, loss)

            # 获取文本输入
            print(batch["input_ids"])
            tokens = batch["input_ids"]
            if args.negative_text is not None:
                tokens += [sd.tokenizer.tokenize(args.negative_text)]
            lengths = [len(t) for t in tokens]
            N = max(lengths)
            tokens = [t + [0] * (N - len(t)) for t in tokens]
            tokens = mx.array(tokens)
            encoder_hidden_states = sd.text_encoder(tokens).last_hidden_state
            if args.n_images > 1:
                encoder_hidden_states = mx.repeat(encoder_hidden_states, args.n_images, axis=0)

            # 图像编码成潜在向量
            start_step = sd.sampler.max_time * args.strength
            # print(batch["pixel_values"])
            print("===============================")
            pixel_values = mx.array(np.array(batch["pixel_values"])[step])
            print(pixel_values.shape)
            # 归一化
            pixel_values = (pixel_values[:, :, :3].astype(mx.float32) / 255) * 2 - 1
            print(pixel_values.shape)
            print(pixel_values[None].shape)
            latents, _ = sd.autoencoder.encode(pixel_values[None])
            latents = mx.broadcast_to(latents, (args.n_images,) + latents.shape[1:])

            noise = mx.random.normal(latents.shape)
            s = sd.sampler.sigmas(mx.array(start_step))
            noisy_latents = (latents + noise * s) * (s.square() + 1).rsqrt()
            for t, t_prev in sd.sampler.timesteps(
                50, start_time=start_step, dtype=sd.dtype
            ):
                x_t_unet = mx.concatenate([noisy_latents] * 2, axis=0)
                t_unet = mx.broadcast_to(t, [len(x_t_unet)])
                eps_pred = sd.unet(x_t_unet, t_unet, encoder_x=encoder_hidden_states)
                # eps_text, eps_neg = eps_pred.split(2)
                # eps_pred = eps_neg + args.cfg * (eps_text - eps_neg)
                x_t_prev = sd.sampler.step(eps_pred, noisy_latents, t, t_prev)

            lvalue, grad = loss_value_and_grad(x_t_prev, x_t_unet)
            losses.append(lvalue.item())

            optimizer.update(sd.unet, grad)
            mx.eval(sd.unet.parameters(), optimizer.state, lvalue)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                progress_bar.update(1)
                global_step += 1

            if (step + 1) % args.gradient_accumulation_steps == 0:
                train_loss = np.mean(losses)

                stop = time.perf_counter()
                print(
                    f"Iter {step + 1}: Train loss {train_loss:.3f}, "
                    f"It/sec {args.gradient_accumulation_steps / (stop - start):.3f}, "
                )
                losses = []
                start = time.perf_counter()

        if global_step >= max_train_steps:
            break


def save_lora_layers(sd, args):
    mx.savez(args.adapter_file, **dict(tree_flatten(sd.unet.trainable_parameters())))

def main():
    parser = argparse.ArgumentParser(description="LoRA or QLoRA finetuning.")
    parser.add_argument("--model", type=str, default="mlx_model")
    parser.add_argument("--n_images", type=int, default=1)
    parser.add_argument("--strength", type=float, default=0.9)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--negative_text", default="")
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
    parser.add_argument("--data_dir", type=str, default="dataset/femme_fatale")
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4, help="Minibatch size.")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--noise_offset", type=float, default=0.1)
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--adapter-file", type=str, default="adapters.npz")
    args = parser.parse_args()
    print(args.n_epochs)

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
