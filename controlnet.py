import torch
from diffusers import ControlNetModel
from controlnet_preprocess import ControlNetPreprocessor
from weights_downloader import WeightsDownloader

CONTROLNET_MODEL_CACHE = "./controlnet-cache"
CONTROLNET_URL = "https://weights.replicate.delivery/default/controlnet/sdxl-cn-canny-depth-softe-pose-qr.tar"


class ControlNet:
    CONTROLNET_MODELS = [
        "none",
        "edge_canny",
        "illusion",
        "depth_leres",
        "depth_midas",
        "soft_edge_pidi",
        "soft_edge_hed",
        "lineart",
        "lineart_anime",
        "openpose",
        # Preprocessors without an XL model yet
        # "straight_edge_mlsd",
        # "face_detector",
        # "content_shuffle",
        # "normal_bae",
        # "segementation_sam",
    ]

    def __init__(self, predictor):
        WeightsDownloader.download_if_not_exists(CONTROLNET_URL, CONTROLNET_MODEL_CACHE)

        self.controlnet_preprocessor = ControlNetPreprocessor(predictor)
        self.models = {
            "canny": self.initialize_controlnet(
                "diffusers/controlnet-canny-sdxl-1.0",
            ),
            "depth": self.initialize_controlnet(
                "diffusers/controlnet-depth-sdxl-1.0-small",
            ),
            "soft_edge": self.initialize_controlnet(
                "SargeZT/controlnet-sd-xl-1.0-softedge-dexined",
            ),
            "openpose": self.initialize_controlnet(
                "thibaud/controlnet-openpose-sdxl-1.0",
            ),
            "illusion": self.initialize_controlnet(
                "monster-labs/control_v1p_sdxl_qrcode_monster",
            ),
        }

    def initialize_controlnet(self, model_name):
        print("Initializing", model_name)
        return ControlNetModel.from_pretrained(
            model_name, cache_dir=CONTROLNET_MODEL_CACHE, torch_dtype=torch.float16
        )

    def get_model(self, controlnet_name):
        if controlnet_name in self.models:
            return self.models[controlnet_name]
        elif controlnet_name.startswith("edge_"):
            return self.models["canny"]
        elif controlnet_name.startswith("depth_"):
            return self.models["depth"]
        elif controlnet_name.startswith("soft_edge") or controlnet_name.startswith(
            "lineart"
        ):
            return self.models["soft_edge"]
        else:
            return None

    def get_models(self, controlnet_names):
        models = [
            self.get_model(controlnet_name) for controlnet_name in controlnet_names
        ]
        return list(filter(None, models))

    def preprocess(self, image, controlnet_name):
        # Illusion model needs no preprocessing
        if controlnet_name == "illusion" or controlnet_name == "none":
            return image

        return self.controlnet_preprocessor.process_image(image, controlnet_name)

    @staticmethod
    def get_controlnet_names():
        return ControlNet.CONTROLNET_MODELS
