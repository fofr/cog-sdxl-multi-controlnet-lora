import torch
from PIL import Image
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


class ControlNetPreprocessor:
    ANNOTATOR_NAMES = [
        "none",
        "edge_canny",
        "depth_leres",
        "depth_midas",
        "soft_edge_pidi",
        "soft_edge_hed",
        "lineart",
        "lineart_anime",
        "openpose",

        # "straight_edge_mlsd",
        # "face_detector",
        # "content_shuffle",
        # "normal_bae",
        # "segementation_sam",
    ]

    def __init__(self, predictor):
        self.annotators = {
            "edge_canny": CannyDetector(),
            "depth_leres": self.initialize_detector(LeresDetector),
            "depth_midas": self.initialize_detector(MidasDetector),
            "soft_edge_pidi": self.initialize_detector(PidiNetDetector),
            "soft_edge_hed": self.initialize_detector(HEDdetector),
            "lineart": self.initialize_detector(LineartDetector),
            "lineart_anime": self.initialize_detector(LineartAnimeDetector),
            "openpose": self.initialize_detector(OpenposeDetector),

            # "straight_edge_mlsd": self.initialize_detector(MLSDdetector),
            # "face_detector": MediapipeFaceDetector(),
            # "content_shuffle": ContentShuffleDetector(),
            # "normal_bae": self.initialize_detector(NormalBaeDetector),
            # "segementation_sam": self.initialize_detector(
            #     SamDetector,
            #     model_name="ybelkada/segment-anything",
            #     subfolder="checkpoints",
            # ),
        }

        torch.device("cuda")

    @staticmethod
    def get_annotator_names():
        return ControlNetPreprocessor.ANNOTATOR_NAMES

    def initialize_detector(
        self, detector_class, model_name="lllyasviel/Annotators", **kwargs
    ):
        print(f"Initializing {detector_class.__name__}")
        return detector_class.from_pretrained(
            model_name,
            cache_dir="model_cache",
            **kwargs,
        )

    def annotators_list(self):
        return list(self.annotators.keys())

    def process_image(self, image, annotator):
        print(f"Processing image with {annotator}")
        return self.annotators[annotator](image)
