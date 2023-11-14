from PIL import Image

OPTIMUM_DIMENSION = 1024
LOWEST_DIMENSION = 512
MAX_DIMENSION = 2048


class SizingStrategy:
    def __init__(self):
        pass

    def get_dimensions(self, image):
        original_width, original_height = image.size
        print(
            f"Original dimensions: Width: {original_width}, Height: {original_height}"
        )
        resized_width, resized_height = self.get_resized_dimensions(
            original_width, original_height
        )
        print(
            f"Dimensions to resize to: Width: {resized_width}, Height: {resized_height}"
        )
        return resized_width, resized_height

    def get_allowed_dimensions(self, base=LOWEST_DIMENSION, max_dim=MAX_DIMENSION):
        """
        Function to generate allowed dimensions optimized around a base up to a max
        """
        allowed_dimensions = []
        for i in range(base, max_dim + 1, 64):
            for j in range(base, max_dim + 1, 64):
                allowed_dimensions.append((i, j))
        return allowed_dimensions

    def get_resized_dimensions(self, width, height):
        allowed_dimensions = self.get_allowed_dimensions()
        aspect_ratio = width / height
        print(f"Aspect Ratio: {aspect_ratio:.2f}")
        # Find the closest allowed dimensions that maintain the aspect ratio
        # and are closest to the optimum dimension
        closest_dimensions = min(
            allowed_dimensions,
            key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
            + abs(dim[0] - OPTIMUM_DIMENSION),
        )
        return closest_dimensions

    def resize_images(self, images, width, height):
        return [
            img.resize((width, height)) if img is not None else None for img in images
        ]

    def open_image(self, image_path):
        return Image.open(str(image_path)) if image_path is not None else None

    def apply(
        self,
        sizing_strategy,
        width,
        height,
        image=None,
        mask=None,
        control_1_image=None,
        control_2_image=None,
        control_3_image=None,
    ):
        image_dict = {
            "input_image": self.open_image(image),
            "mask_image": self.open_image(mask),
            "controlnet_1_image": self.open_image(control_1_image),
            "controlnet_2_image": self.open_image(control_2_image),
            "controlnet_3_image": self.open_image(control_3_image),
        }

        if sizing_strategy in image_dict:
            print(f"Resizing based on {sizing_strategy}")
            width, height = self.get_dimensions(image_dict[sizing_strategy])
        else:
            print("Using given dimensions")

        resized_images = self.resize_images(
            list(image_dict.values()),
            width,
            height,
        )

        return width, height, resized_images
