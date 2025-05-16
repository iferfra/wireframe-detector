import argparse
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

from models.wireframe_net import WireframeNet


def load_image(path):
    img = Image.open(path).convert("RGB")
    tensor = (
        torch.from_numpy(__import__("numpy").array(img)).permute(2, 0, 1).float()
        / 255.0
    )
    return img, tensor.unsqueeze(0)


def draw_wireframe(img, results, output_path):
    keypoints = results["points"].squeeze(0).detach().cpu().numpy()
    fig2, axs2 = plt.subplots(1, 2, dpi=500)
    l_segments0 = results["line_segments"]
    img_plot = np.transpose(img[0].detach().cpu().numpy(), (1, 2, 0))
    img_plot = img_plot[..., [2, 1, 0]]
    img_plot = cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB)
    axs2[0].imshow(img_plot)
    axs2[0].axis("off")
    axs2[1].imshow(img_plot)
    axs2[1].axis("off")

    for i in range(len(l_segments0)):
        line0 = matplotlib.lines.Line2D(
            (l_segments0[i, 0, 0], l_segments0[i, 1, 0]),
            (l_segments0[i, 0, 1], l_segments0[i, 1, 1]),
            zorder=1,
            c="orange",
            linewidth=1,
            alpha=1.0,
        )
        axs2[0].add_line(line0)
    pts0 = l_segments0.reshape(-1, 2)
    axs2[0].scatter(
        pts0[:, 0], pts0[:, 1], c="cyan", s=1, linewidths=1, zorder=2, alpha=1.0
    )

    axs2[1].scatter(
        keypoints[:, 0], keypoints[:, 1], c="red", s=0.2, zorder=2, alpha=1.0
    )

    axs2[0].set_title("Wireframe")
    axs2[1].set_title("Keypoints")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

    print("Results saved to {}".format(output_path))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="outputs/results.png")
    args = parser.parse_args()

    Path("outputs").mkdir(parents=True, exist_ok=True)

    config = {"weights": args.checkpoint, "size": [640, 640]}
    device = torch.device(args.device)

    model = WireframeNet(config)
    model.eval().to(device)

    orig, img_t = load_image(args.image)
    img_t = img_t.to(device)
    result = model({"image": img_t})

    draw_wireframe(img_t, result, args.output)


if __name__ == "__main__":
    main()
