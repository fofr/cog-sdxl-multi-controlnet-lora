import os
import time
from typing import List, Optional

import numpy as np
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
)

from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from transformers import CLIPImageProcessor
from weights_downloader import WeightsDownloader
from weights_manager import WeightsManager
from controlnet import ControlNet
from sizing_strategy import SizingStrategy


SDXL_MODEL_CACHE = "./sdxl-cache"
REFINER_MODEL_CACHE = "./refiner-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"
REFINER_URL = (
    "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
)
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"


class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}


class Predictor(BasePredictor):
    def load_trained_weights(self, weights, pipe):
        self.weights_manager.load_trained_weights(weights, pipe)

    def build_controlnet_pipeline(self, pipeline_class, controlnet_models):
        pipe = pipeline_class.from_pretrained(
            SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
            controlnet=self.controlnet.get_models(controlnet_models),
        )

        pipe.to("cuda")

        return pipe

    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""

        start = time.time()
        self.sizing_strategy = SizingStrategy()
        self.weights_manager = WeightsManager(self)
        self.tuned_model = False
        self.tuned_weights = None
        if str(weights) == "weights":
            weights = None

        print("Loading safety checker...")
        WeightsDownloader.download_if_not_exists(SAFETY_URL, SAFETY_CACHE)

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        WeightsDownloader.download_if_not_exists(SDXL_URL, SDXL_MODEL_CACHE)

        print("Loading sdxl txt2img pipeline...")
        self.txt2img_pipe = DiffusionPipeline.from_pretrained(
            SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.is_lora = False
        if weights or os.path.exists("./trained-model"):
            self.load_trained_weights(weights, self.txt2img_pipe)

        self.txt2img_pipe.to("cuda")

        print("Loading SDXL img2img pipeline...")
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.img2img_pipe.to("cuda")

        print("Loading SDXL inpaint pipeline...")
        self.inpaint_pipe = StableDiffusionXLInpaintPipeline(
            vae=self.txt2img_pipe.vae,
            text_encoder=self.txt2img_pipe.text_encoder,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            tokenizer=self.txt2img_pipe.tokenizer,
            tokenizer_2=self.txt2img_pipe.tokenizer_2,
            unet=self.txt2img_pipe.unet,
            scheduler=self.txt2img_pipe.scheduler,
        )
        self.inpaint_pipe.to("cuda")

        print("Loading SDXL refiner pipeline...")
        # FIXME(ja): should the vae/text_encoder_2 be loaded from SDXL always?
        #            - in the case of fine-tuned SDXL should we still?
        # FIXME(ja): if the answer to above is use VAE/Text_Encoder_2 from fine-tune
        #            what does this imply about lora + refiner? does the refiner need to know about

        WeightsDownloader.download_if_not_exists(REFINER_URL, REFINER_MODEL_CACHE)

        print("Loading refiner pipeline...")
        self.refiner = DiffusionPipeline.from_pretrained(
            REFINER_MODEL_CACHE,
            text_encoder_2=self.txt2img_pipe.text_encoder_2,
            vae=self.txt2img_pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")

        self.controlnet = ControlNet(self)

        print("setup took: ", time.time() - start)

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Negative Prompt",
            default="",
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        mask: Path = Input(
            description="Input mask for inpaint mode. Black areas will be preserved, white areas will be inpainted.",
            default=None,
        ),
        width: int = Input(
            description="Width of output image",
            default=768,
        ),
        height: int = Input(
            description="Height of output image",
            default=768,
        ),
        sizing_strategy: str = Input(
            description="Decide how to resize images – use width/height, resize based on input image or control image",
            choices=[
                "width_height",
                "input_image",
                "controlnet_1_image",
                "controlnet_2_image",
                "controlnet_3_image",
                "mask_image",
            ],
            default="width_height",
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=30
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        prompt_strength: float = Input(
            description="Prompt strength when using img2img / inpaint. 1.0 corresponds to full destruction of information in image",
            ge=0.0,
            le=1.0,
            default=0.8,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        refine: str = Input(
            description="Which refine style to use",
            choices=["no_refiner", "base_image_refiner"],
            default="no_refiner",
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine, defaults to num_inference_steps",
            default=None,
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),
        lora_scale: float = Input(
            description="LoRA additive scale. Only applicable on trained models.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        lora_weights: str = Input(
            description="Replicate LoRA weights to use. Leave blank to use the default weights.",
            default=None,
        ),
        disable_safety_checker: bool = Input(
            description="Disable safety checker for generated images. This feature is only available through the API. See [https://replicate.com/docs/how-does-replicate-work#safety](https://replicate.com/docs/how-does-replicate-work#safety)",
            default=False,
        ),
        controlnet_1: str = Input(
            description="Controlnet",
            choices=ControlNet.CONTROLNET_MODELS,
            default="none",
        ),
        controlnet_1_image: Path = Input(
            description="Input image for first controlnet",
            default=None,
        ),
        controlnet_1_conditioning_scale: float = Input(
            description="How strong the controlnet conditioning is",
            ge=0.0,
            le=4.0,
            default=0.75,
        ),
        controlnet_1_start: float = Input(
            description="When controlnet conditioning starts",
            ge=0.0,
            le=1.0,
            default=0.0,
        ),
        controlnet_1_end: float = Input(
            description="When controlnet conditioning ends",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        controlnet_2: str = Input(
            description="Controlnet",
            choices=ControlNet.CONTROLNET_MODELS,
            default="none",
        ),
        controlnet_2_image: Path = Input(
            description="Input image for second controlnet",
            default=None,
        ),
        controlnet_2_conditioning_scale: float = Input(
            description="How strong the controlnet conditioning is",
            ge=0.0,
            le=4.0,
            default=0.75,
        ),
        controlnet_2_start: float = Input(
            description="When controlnet conditioning starts",
            ge=0.0,
            le=1.0,
            default=0.0,
        ),
        controlnet_2_end: float = Input(
            description="When controlnet conditioning ends",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
        controlnet_3: str = Input(
            description="Controlnet",
            choices=ControlNet.CONTROLNET_MODELS,
            default="none",
        ),
        controlnet_3_image: Path = Input(
            description="Input image for third controlnet",
            default=None,
        ),
        controlnet_3_conditioning_scale: float = Input(
            description="How strong the controlnet conditioning is",
            ge=0.0,
            le=4.0,
            default=0.75,
        ),
        controlnet_3_start: float = Input(
            description="When controlnet conditioning starts",
            ge=0.0,
            le=1.0,
            default=0.0,
        ),
        controlnet_3_end: float = Input(
            description="When controlnet conditioning ends",
            ge=0.0,
            le=1.0,
            default=1.0,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model."""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        (
            width,
            height,
            resized_images,
        ) = self.sizing_strategy.apply(
            sizing_strategy,
            width,
            height,
            image,
            mask,
            controlnet_1_image,
            controlnet_2_image,
            controlnet_3_image,
        )

        [
            image,
            mask,
            controlnet_1_image,
            controlnet_2_image,
            controlnet_3_image,
        ] = resized_images

        if lora_weights:
            self.load_trained_weights(lora_weights, self.txt2img_pipe)

        # OOMs can leave vae in bad state
        if self.txt2img_pipe.vae.dtype == torch.float32:
            self.txt2img_pipe.vae.to(dtype=torch.float16)

        sdxl_kwargs = {}
        if self.tuned_model:
            # consistency with fine-tuning API
            for k, v in self.token_map.items():
                prompt = prompt.replace(k, v)
        print(f"Prompt: {prompt}")

        inpainting = image and mask
        img2img = image and not mask
        controlnet = (
            controlnet_1 != "none" or controlnet_2 != "none" or controlnet_3 != "none"
        )

        controlnet_args = {}
        control_images = []
        if controlnet:
            controlnet_conditioning_scales = []
            control_guidance_start = []
            control_guidance_end = []

            controlnets = [
                (
                    controlnet_1,
                    controlnet_1_conditioning_scale,
                    controlnet_1_start,
                    controlnet_1_end,
                    controlnet_1_image,
                ),
                (
                    controlnet_2,
                    controlnet_2_conditioning_scale,
                    controlnet_2_start,
                    controlnet_2_end,
                    controlnet_2_image,
                ),
                (
                    controlnet_3,
                    controlnet_3_conditioning_scale,
                    controlnet_3_start,
                    controlnet_3_end,
                    controlnet_3_image,
                ),
            ]

            for controlnet in controlnets:
                if controlnet[0] != "none":
                    controlnet_conditioning_scales.append(controlnet[1])
                    control_guidance_start.append(controlnet[2])
                    control_guidance_end.append(controlnet[3])
                    annotated_image = self.controlnet.preprocess(
                        controlnet[4], controlnet[0]
                    )
                    control_images.append(annotated_image)

            controlnet_args = {
                "controlnet_conditioning_scale": controlnet_conditioning_scales,
                "control_guidance_start": control_guidance_start,
                "control_guidance_end": control_guidance_end,
            }

            if inpainting:
                controlnet_args["control_image"] = control_images
                pipe = self.build_controlnet_pipeline(
                    StableDiffusionXLControlNetInpaintPipeline,
                    [controlnet[0] for controlnet in controlnets],
                )
            elif img2img:
                controlnet_args["control_image"] = control_images
                pipe = self.build_controlnet_pipeline(
                    StableDiffusionXLControlNetImg2ImgPipeline,
                    [controlnet[0] for controlnet in controlnets],
                )
            else:
                controlnet_args["image"] = control_images
                pipe = self.build_controlnet_pipeline(
                    StableDiffusionXLControlNetPipeline,
                    [controlnet[0] for controlnet in controlnets],
                )

        elif inpainting:
            pipe = self.inpaint_pipe
        elif img2img:
            pipe = self.img2img_pipe
        else:
            pipe = self.txt2img_pipe

        if inpainting:
            sdxl_kwargs["image"] = image
            sdxl_kwargs["mask_image"] = mask
            sdxl_kwargs["strength"] = prompt_strength
        elif img2img:
            sdxl_kwargs["image"] = image
            sdxl_kwargs["strength"] = prompt_strength

        if refine == "base_image_refiner":
            sdxl_kwargs["output_type"] = "latent"

        if not apply_watermark:
            # toggles watermark for this prediction
            watermark_cache = pipe.watermark
            pipe.watermark = None
            self.refiner.watermark = None

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "width": width,
            "height": height,
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        if self.is_lora:
            sdxl_kwargs["cross_attention_kwargs"] = {"scale": lora_scale}

        output = pipe(**common_args, **sdxl_kwargs, **controlnet_args)

        if refine == "base_image_refiner":
            refiner_kwargs = {
                "image": output.images,
            }

            common_args_without_dimensions = {
                k: v for k, v in common_args.items() if k not in ["width", "height"]
            }

            if refine == "base_image_refiner" and refine_steps:
                common_args_without_dimensions["num_inference_steps"] = refine_steps

            output = self.refiner(**common_args_without_dimensions, **refiner_kwargs)

        if not apply_watermark:
            pipe.watermark = watermark_cache
            self.refiner.watermark = watermark_cache

        if not disable_safety_checker:
            _, has_nsfw_content = self.run_safety_checker(output.images)

        output_paths = []

        if controlnet:
            for i, image in enumerate(control_images):
                output_path = f"/tmp/control-{i}.png"
                image.save(output_path)
                output_paths.append(Path(output_path))

        for i, image in enumerate(output.images):
            if not disable_safety_checker:
                if has_nsfw_content[i]:
                    print(f"NSFW content detected in image {i}")
                    continue
            output_path = f"/tmp/out-{i}.png"
            image.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
