import torch
from pathlib import *
import segmentation_models_pytorch as smp


# =====================================================
#    Extracting encoder from a Pre-trained U-Net Model
# =====================================================


def main():
    dir_path = Path("C:\\Users\\aaron.l\\Documents\\FeatureRegression\\model_param")
    model_path = dir_path / "unet_trained.pth"

    model = torch.load(model_path, map_location=torch.device("cpu"))
    print("type of model: ", type(model))
    for name, param in model.items():
        print(
            f"Layer: {name} | Size: {param.size()} | Requires Grad: {param.requires_grad}"
        )

    model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1)
    new_encoder = model.encoder
    encoder_save_path = dir_path / "encoder.pth"
    torch.save(new_encoder.state_dict(), encoder_save_path)


if __name__ == "__main__":
    main()
