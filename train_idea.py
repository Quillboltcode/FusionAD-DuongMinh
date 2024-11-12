import argparse

import os
import torch
import wandb


import numpy as np
from itertools import chain

from tqdm import tqdm, trange
import torchvision.transforms as T

from models.Acmf import ACMF
# from models.ad_models import FeatureExtractors
# from models.feature_transfer_nets import FeatureProjectionMLP, FeatureProjectionMLP_big
from dataset2D import get_dataloader
from models.idea2 import MultiModalNet
from utils.metrics_utils import calculate_au_pro
from utils.general_utils import set_seeds, SquarePad


def set_seeds(sid=115):
    np.random.seed(sid)

    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def train(args):
    set_seeds()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f"ACMF_module{args.person}{args.unique_id}.pth"


    wandb.init(project="AD", name=model_name)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    common = T.Compose(
        [
            SquarePad(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Dataloader.
    train_loader = get_dataloader(
        os.path.join(args.dataset_path, args.class_name, "normal"), common, common, args.batch_size, 16, True)

    # Feature extractors.
    model = MultiModalNet(feature_dim=768)

    # Model instantiation.

    optimizer = torch.optim.Adam(params=chain(model.illumination_net.parameters(),
                                              model.feature_enhancement.parameters(),
                                              model.feature_prediction.parameters(),)
                                )

    model.to(device)

    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)
    loss_fn_enhancement = torch.nn.MSELoss()

    for epoch in trange(
        args.epochs_no, desc=f"MutiModalNet.{args.class_name}"
    ):
        model.train()
        epoch_cos_sim = []
        for i, (well_image, lowlight) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs_no}")
        ):
            well_image, lowlight = well_image.to(device), lowlight.to(device)            
            # if args.batch_size == 1:               
            well_lit_features, enhanced_features, predicted_features, predicted_upsampled_features, well_lit_upsampled_features = model(lowlight, well_image)
            
            loss_enhacement = loss_fn_enhancement(enhanced_features, well_lit_features)

            
            # -------------------------------------------------
            mask = (predicted_upsampled_features.sum(axis=-1) == 0)
            # loss = 1 - \
            #     metric(transfer_features[~low_light_mask],
            #            images[~low_light_mask]).mean()

            loss_predicton = 1 - metric(predicted_upsampled_features[~mask],well_lit_upsampled_features[~mask]).mean()
            # loss = 1 - metric(images, transfer_features).mean()
            #-------------------------------------------------
            # 1. la 2 loss, w-t, r-t
            # 2. using well light mask
            # 3. dung 2 mlp
            #---------------------------------------------------
            # epoch_cos_sim.append(loss.item())
            loss = loss_enhacement + loss_predicton
            if not torch.isnan(loss_enhacement) and not torch.isinf(loss_predicton):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        wandb.log(
            {
                "Epoch": epoch + 1,
                "Loss": np.mean(epoch_cos_sim),
            }
        )
        if not os.path.exists(args.checkpoint_folder):
            os.mkdir(args.checkpoint_folder)
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                MultiModalNet.state_dict(),
                f"{args.checkpoint_folder}/{args.class_name}/{model_name}",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Crossmodal Feature Networks (FADs) on a dataset."
    )

    parser.add_argument(
        "--dataset_path", default="data", type=str, help="Dataset path."
    )

    parser.add_argument(
        "--checkpoint_folder",
        default="checkpoints",
        type=str,
        help="Where to save the model checkpoints.",
    )

    parser.add_argument(
        "--class_name",
        default="medicine_pack",
        type=str,
        choices=[
            "small_battery",
            "screws_toys",
            "furniture",
            "tripod_plugs",
            "water_cups",
            "keys",
            "pens",
            "locks",
            "screwdrivers",
            "charging_cords",
            "pencil_cords",
            "water_cans",
            "pills",
            "locks",
            "medicine_pack",
            "small_bottles",
            "metal_plate",
            "usb_connector_board"
        ],
        help="Category name.",
    )

    parser.add_argument(
        "--epochs_no", default=20, type=int, help="Number of epochs to train the FADs."
    )

    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch dimension. Usually 16 is around the max.",
    )

    parser.add_argument(
        "--save_interval",
        default=5,
        type=int,
        help="Number of epochs to train the FADs.",
    )

    parser.add_argument("--unique_id", type=str, default="v2theta+",
                        help="A unique identifier for the checkpoint (e.g., experiment ID)")

    parser.add_argument("--person", default="DuongMinh" ,type=str,
                        help="Name or initials of the person saving the checkpoint")

    args = parser.parse_args()
    train(args)
