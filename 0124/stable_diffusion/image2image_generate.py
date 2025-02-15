import argparse
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from lora import inject_lora, LoraLayer
from stable_diffusion import StableDiffusion
from tqdm import tqdm
from PIL import Image

def main():
    parser = argparse.ArgumentParser(
        description="Generate images from an image and a textual prompt using stable diffusion with lora"
    )
    parser.add_argument("image")
    parser.add_argument("prompt")
    parser.add_argument("--no-float16", dest="float16", action="store_false")
    parser.add_argument("--adapter-file", type=str, default="adapters.npz")
    parser.add_argument("--strength", type=float, default=0.6)
    parser.add_argument("--n_images", type=int, default=1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--output", default="out.png")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--n_rows", type=int, default=1)
    parser.add_argument("--decoding_batch_size", type=int, default=1)
    args = parser.parse_args()

    sd = StableDiffusion("stabilityai/stable-diffusion-2-1-base", float16=args.float16)

    for name,layer in sd.unet.named_modules():
        name_cols=name.split('.')
        filter_names=['query_proj','key_proj','value_proj']
        if any(n in name_cols for n in filter_names) and isinstance(layer,nn.Linear):
            inject_lora(sd.unet,name,layer)

    sd.unet.load_weights(args.adapter_file, strict=False)

    for name,layer in sd.unet.named_modules():
        name_cols=name.split('.')

        if isinstance(layer, LoraLayer):
            children=name_cols[:-1]
            cur_layer=sd.unet
            for i in range(0, len(children)-1, 2):
                child=children[i]
                cur_layer=getattr(cur_layer,child)[int(children[i+1])]
            lora_weight=(layer.lora_a@layer.lora_b)*layer.alpha/layer.r
            layer.raw_linear.weight=layer.raw_linear.weight + lora_weight.T
            setattr(cur_layer,name_cols[-1],layer.raw_linear)

    # Read the image
    img = Image.open(args.image)

    # Make sure image shape is divisible by 64
    W, H = (dim - dim % 64 for dim in (img.width, img.height))
    if W != img.width or H != img.height:
        print(f"Warning: image shape is not divisible by 64, downsampling to {W}x{H}")
        img = img.resize((W, H), Image.NEAREST)  # use desired downsampling filter

    img = mx.array(np.array(img))
    img = (img[:, :, :3].astype(mx.float32) / 255) * 2 - 1

    latents = sd.generate_latents_from_image(
        img,
        args.prompt,
        strength=args.strength,
        n_images=args.n_images,
        cfg_weight=args.cfg,
        num_steps=args.steps,
        negative_text=args.negative_prompt,
        seed=args.seed,
    )
    for x_t in tqdm(latents, total=int(args.steps * args.strength)):
        mx.eval(x_t)

    decoded = []
    for i in tqdm(range(0, args.n_images, args.decoding_batch_size)):
        decoded.append(sd.decode(x_t[i : i + args.decoding_batch_size]))
        mx.eval(decoded[-1])
    peak_mem_overall = mx.metal.get_peak_memory() / 1024**3

    # Arrange them on a grid
    x = mx.concatenate(decoded, axis=0)
    x = mx.pad(x, [(0, 0), (8, 8), (8, 8), (0, 0)])
    B, H, W, C = x.shape
    x = x.reshape(args.n_rows, B // args.n_rows, H, W, C).transpose(0, 2, 1, 3, 4)
    x = x.reshape(args.n_rows * H, B // args.n_rows * W, C)
    x = (x * 255).astype(mx.uint8)

    # Save them to disc
    im = Image.fromarray(np.array(x))
    im.save(args.output)

    print(f"Peak memory used overall:      {peak_mem_overall:.3f}GB")

if __name__ == "__main__":
    main()