import torch
from weights_downloader import WeightsDownloader
from controlnet_aux import (
    HEDdetector,
    MidasDetector,
    OpenposeDetector,
    PidiNetDetector,
    LineartDetector,
    LineartAnimeDetector,
    CannyDetector,
    LeresDetector,
)

CONTROLNET_PREPROCESSOR_MODEL_CACHE = "./controlnet-preprocessor-cache"
CONTROLNET_PREPROCESSOR_URL = "https://weights.replicate.delivery/default/controlnet/cn-preprocess-leres-midas-pidi-hed-lineart-openpose.tar"


class ControlNetPreprocessor:
    ANNOTATOR_CLASSES = {
        "none": None,
        "edge_canny": CannyDetector,
        "depth_leres": LeresDetector,
        "depth_midas": MidasDetector,
        "soft_edge_pidi": PidiNetDetector,
        "soft_edge_hed": HEDdetector,
        "lineart": LineartDetector,
        "lineart_anime": LineartAnimeDetector,
        "openpose": OpenposeDetector,
        # "straight_edge_mlsd": None,
        # "face_detector": None,
        # "content_shuffle": None,
        # "normal_bae": None,
        # "segementation_sam": None,
    }

    ANNOTATOR_NAMES = list(ANNOTATOR_CLASSES.keys())

    def __init__(self, predictor):
        WeightsDownloader.download_if_not_exists(
            CONTROLNET_PREPROCESSOR_URL, CONTROLNET_PREPROCESSOR_MODEL_CACHE
        )

        self.annotators = {}
        self.predictor = predictor

        torch.device("cuda")

    @staticmethod
    def get_annotator_names():
        return ControlNetPreprocessor.ANNOTATOR_NAMES

    def initialize_detector(
        self, detector_class, model_name="lllyasviel/Annotators", **kwargs
    ):
        print(f"Initializing {detector_class.__name__}")
        if hasattr(detector_class, 'from_pretrained'):
            return detector_class.from_pretrained(
                model_name,
                cache_dir=CONTROLNET_PREPROCESSOR_MODEL_CACHE,
                **kwargs,
            )
        else:
            return detector_class(**kwargs)

    def annotators_list(self):
        return list(self.annotators.keys())

    def process_image(self, image, annotator):
        print(f"Processing image with {annotator}")
        if annotator not in self.annotators:
            self.annotators[annotator] = self.initialize_detector(
                self.ANNOTATOR_CLASSES[annotator]
            )
        return self.annotators[annotator](image)
