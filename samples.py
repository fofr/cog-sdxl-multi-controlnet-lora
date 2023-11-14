"""
A handy utility for verifying SDXL image generation locally.
To set up, first run a local cog server using:
   cog run -p 5000 python -m cog.server.http
Then, in a separate terminal, generate samples
   python samples.py
"""


import base64
import os
import sys
import requests
import glob


def gen(output_fn, **kwargs):
    if glob.glob(f"{output_fn}*"):
        return

    print("Generating", output_fn)
    url = "http://localhost:5000/predictions"
    response = requests.post(url, json={"input": kwargs})
    data = response.json()

    try:
        for i, datauri in enumerate(data["output"]):
            base64_encoded_data = datauri.split(",")[1]
            decoded_data = base64.b64decode(base64_encoded_data)
            with open(
                f"{output_fn.rsplit('.', 1)[0]}_{i}.{output_fn.rsplit('.', 1)[1]}", "wb"
            ) as f:
                f.write(decoded_data)
    except:
        print("Error!")
        print("input:", kwargs)
        print(data["logs"])
        sys.exit(1)


def main():
    CONTROLNET_MODELS = [
        "none",
        "edge_canny",
        "edge_pidi",
        "illusion",
        "depth_leres",
        "depth_midas",
        "soft_edge_hed",
        "lineart",
        "lineart_anime",
        "openpose",
    ]

    gen(
        "sample.txt2img.png",
        prompt="A studio portrait photo of a cat",
        num_inference_steps=25,
        guidance_scale=7,
        negative_prompt="ugly, soft, blurry, out of focus, low quality, garish, distorted, disfigured",
        seed=1000,
        width=1024,
        height=1024,
    )

    for c in CONTROLNET_MODELS:
        gen(
            f"sample.{c}.txt2img.png",
            prompt="A studio portrait photo of a cat",
            num_inference_steps=25,
            controlnet_1=c,
            controlnet_1_image="https://pbxt.replicate.delivery/YXbcLudoHBIYHV6L0HbcTx5iRzLFMwygLr3vhGpZI35caXbE/out-0.png",
            seed=1000,
            width=768,
            height=768,
        )


if __name__ == "__main__":
    main()
