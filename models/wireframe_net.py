import kornia
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.functional import grid_sample
from torchvision import transforms

from models import project_root
from models.base_model import BaseModel
from models.hawp_utils import WireframeGraph


class UNET(nn.Module):

    def __init__(self):
        super().__init__()

        self.unet = kornia.feature.DISK.from_pretrained("depth")

    def forward(self, x):
        prediction = self.unet.heatmap_and_dense_descriptors(x)
        return prediction


# Heat Map and Descriptors
class DiskDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_heatmap_1 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=1, bias=True)

        self.conv_descriptors_1 = torch.nn.Conv2d(
            128, 128, 3, stride=1, padding=1, bias=True
        )
        self.conv_descriptors_2 = torch.nn.Conv2d(
            128, 128, 3, stride=1, padding=1, bias=True
        )
        self.conv_descriptors_3 = torch.nn.Conv2d(
            128, 128, 3, stride=1, padding=1, bias=True
        )
        self.conv_descriptors_4 = torch.nn.Conv2d(
            128, 128, 1, stride=1, padding=0, bias=True
        )

        self.activation = torch.nn.LeakyReLU(negative_slope=0.01)

        self.batch_norm_descriptors = nn.BatchNorm2d(128)

    def forward(self, input_batch):

        heatmap = input_batch[0]
        descriptors = input_batch[1]

        heatmap = self.conv_heatmap_1(heatmap)

        descriptors = self.conv_descriptors_1(descriptors)
        descriptors = self.batch_norm_descriptors(descriptors)
        descriptors = self.activation(descriptors)
        descriptors = self.conv_descriptors_2(descriptors)
        descriptors = self.activation(descriptors)
        descriptors = self.conv_descriptors_3(descriptors)
        descriptors = self.activation(descriptors)
        descriptors = self.conv_descriptors_4(descriptors)

        output = torch.cat((heatmap, descriptors), dim=-3)

        return output


class HawpDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.attention_conv_1d = torch.nn.Conv2d(
            128, 128, 1, stride=1, padding=0, bias=True
        )

        # Feature decoder block
        self.conv_1_features = torch.nn.Conv2d(
            128, 256, 3, stride=2, padding=1, bias=True
        )
        self.batch_norm_features = torch.nn.BatchNorm2d(256)
        self.conv_2_features = torch.nn.Conv2d(
            256, 256, 3, stride=2, padding=1, bias=True
        )
        self.conv_3_features = torch.nn.Conv2d(
            256, 256, 1, stride=1, padding=0, bias=True
        )

        # AFM decoder block
        self.conv_1_afm = torch.nn.Conv2d(256, 128, 3, stride=1, padding=1, bias=True)
        self.batch_norm_afm = torch.nn.BatchNorm2d(128)
        self.conv_2_afm = torch.nn.Conv2d(128, 32, 1, stride=1, padding=0, bias=True)
        self.conv_3_afm = torch.nn.Conv2d(32, 5, 1, stride=1, padding=0, bias=True)

        # Junctions decoder block
        self.conv_1_junctions = torch.nn.Conv2d(
            256, 128, 1, stride=1, padding=0, bias=True
        )
        self.batch_norm_junctions = torch.nn.BatchNorm2d(128)
        self.conv_2_junctions = torch.nn.Conv2d(
            128, 32, 1, stride=1, padding=0, bias=True
        )
        self.conv_3_junctions = torch.nn.Conv2d(
            32, 4, 1, stride=1, padding=0, bias=True
        )

        # LOI features decoder block
        self.conv_loi_features = torch.nn.Conv2d(
            256, 128, 1, stride=1, padding=0, bias=True
        )
        self.conv_loi_features_thin = torch.nn.Conv2d(
            256, 4, 1, stride=1, padding=0, bias=True
        )
        self.conv_loi_features_aux = torch.nn.Conv2d(
            256, 4, 1, stride=1, padding=0, bias=True
        )
        self.activation = torch.nn.ReLU()

    def forward(self, input_batch):

        attention = self.attention_conv_1d(input_batch)
        mask_avg = F.adaptive_avg_pool2d(attention, (1, 1))
        mask = F.sigmoid(mask_avg)

        x = input_batch * mask

        x = self.conv_1_features(x)
        x = self.batch_norm_features(x)
        x = self.activation(x)
        x = self.conv_2_features(x)
        x = self.activation(x)
        x = self.conv_3_features(x)
        features = self.activation(x)

        x = self.conv_1_afm(features)
        x = self.batch_norm_afm(x)
        x = self.activation(x)
        x = self.conv_2_afm(x)
        x = self.activation(x)
        afm = self.conv_3_afm(x)

        x = self.conv_1_junctions(features)
        x = self.batch_norm_junctions(x)
        x = self.activation(x)
        x = self.conv_2_junctions(x)
        x = self.activation(x)
        junctions = self.conv_3_junctions(x)

        loi_features = self.conv_loi_features(features)
        loi_features_thin = self.conv_loi_features_thin(features)
        loi_features_aux = self.conv_loi_features_aux(features)

        return {
            "features": features,
            "afm": afm,
            "junctions": junctions,
            "loi_features": loi_features,
            "loi_features_thin": loi_features_thin,
            "loi_features_aux": loi_features_aux,
        }


class DiskHeaders(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor, juncs_pred_hawp, device):

        heatmap = input_tensor[:, 0:1, :, :]
        keypoints = kornia.feature.disk.detector.heatmap_to_keypoints(
            heatmap,
            n=None,
            window_size=15,
            score_threshold=0.99,
        )

        if juncs_pred_hawp.dim() == 2:
            juncs_pred_hawp = juncs_pred_hawp.unsqueeze(0)

        keypoints_final = [keypoints[i].xys for i in range(input_tensor.shape[0])]
        keypoints_final = torch.stack(keypoints_final)

        if keypoints_final.dim() == 2:
            keypoints_final = keypoints_final.unsqueeze(0)

        keypoints_final = torch.cat(
            (juncs_pred_hawp.type(torch.int), keypoints_final), dim=-2
        )

        feature_map = input_tensor[:, 1:129, :, :]

        descriptors = []
        for i in range(input_tensor.shape[0]):

            x, y = keypoints_final[i].T
            desc = feature_map[i, :, y, x].T
            desc = F.normalize(desc, dim=-1)
            descriptors.append(desc)

        descriptors = torch.stack(descriptors, dim=-3)

        features = []
        for i in range(input_tensor.shape[0]):
            features.append(
                keypoints[i].merge_with_descriptors(input_tensor[:, 1:129, :, :][i])
            )

        scores_keypoints = [
            features[i].detection_scores for i in range(input_tensor.shape[0])
        ]
        scores_keypoints = [i / i.max() for i in scores_keypoints]
        scores_keypoints = torch.stack(scores_keypoints)

        scores_juncs_pred = torch.ones(
            input_tensor.shape[0], juncs_pred_hawp.shape[-2]
        ).to(device)
        scores = torch.cat((scores_juncs_pred, scores_keypoints), dim=-1)

        return keypoints_final, descriptors, scores


def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask


class HawpHeaders(nn.Module):
    def __init__(self, size):
        super().__init__()

        self.size = size

        self.fc1 = nn.Conv2d(256, 128, kernel_size=1, stride=1)
        self.fc3 = nn.Conv2d(256, 4, kernel_size=1, stride=1)
        self.fc4 = nn.Conv2d(256, 4, kernel_size=1, stride=1)
        self.fc2 = nn.Sequential(
            nn.Linear(496, 1024, bias=True),
            nn.ReLU(),
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(),
            nn.Linear(1024, 1024, bias=True),
        )
        self.fc2_head = nn.Linear(1024, 2, bias=True)
        self.fc2_res = nn.Sequential(nn.Linear(240, 1024, bias=True), nn.ReLU())

    def wireframe_matcher(
        self, juncs_pred, lines_pred, is_train=False, is_shuffle=False
    ):
        cost1 = torch.sum((lines_pred[:, :2] - juncs_pred[:, None]) ** 2, dim=-1)
        cost2 = torch.sum((lines_pred[:, 2:] - juncs_pred[:, None]) ** 2, dim=-1)

        dis1, idx_junc_to_end1 = cost1.min(dim=0)
        dis2, idx_junc_to_end2 = cost2.min(dim=0)

        idx_junc_to_end_min = torch.min(idx_junc_to_end1, idx_junc_to_end2)
        idx_junc_to_end_max = torch.max(idx_junc_to_end1, idx_junc_to_end2)

        iskeep = idx_junc_to_end_min < idx_junc_to_end_max
        if 10.0 > 0:
            iskeep *= (dis1 < 10.0) * (dis2 < 10.0)

        idx_lines_for_junctions = torch.stack(
            (idx_junc_to_end_min[iskeep], idx_junc_to_end_max[iskeep]), dim=1
        )

        if is_shuffle:
            cost_atoi_argsort = torch.randperm(iskeep.sum(), device=juncs_pred.device)
        else:
            cost_atoi = torch.min(
                torch.sum(
                    (
                        juncs_pred[idx_lines_for_junctions].reshape(-1, 4)
                        - lines_pred[iskeep]
                    )
                    ** 2,
                    dim=-1,
                ),
                torch.sum(
                    (
                        juncs_pred[idx_lines_for_junctions].reshape(-1, 4)
                        - lines_pred[iskeep][:, [2, 3, 0, 1]]
                    )
                    ** 2,
                    dim=-1,
                ),
            )

            cost_atoi_argsort = cost_atoi.argsort(descending=True)

        lines_pred_kept = lines_pred[iskeep][cost_atoi_argsort]
        idx_lines_for_junctions = idx_lines_for_junctions[cost_atoi_argsort]

        _, perm = np.unique(
            idx_lines_for_junctions.cpu().numpy(), return_index=True, axis=0
        )

        idx_lines_for_junctions = idx_lines_for_junctions[perm]
        lines_init = lines_pred_kept[perm]

        if is_train:
            idx_lines_for_junctions_mirror = torch.cat(
                (
                    idx_lines_for_junctions[:, 1, None],
                    idx_lines_for_junctions[:, 0, None],
                ),
                dim=1,
            )
            idx_lines_for_junctions = torch.cat(
                (idx_lines_for_junctions, idx_lines_for_junctions_mirror)
            )

        lines_adjusted = juncs_pred[idx_lines_for_junctions].reshape(-1, 4)

        return lines_adjusted, lines_init, perm, idx_lines_for_junctions

    def hafm_decoding(self, md_maps, dis_maps, residual_maps, flatten=True):

        device = md_maps.device
        scale = 2

        batch_size, _, height, width = md_maps.shape
        _y = torch.arange(0, height, device=device).float()
        _x = torch.arange(0, width, device=device).float()

        y0, x0 = torch.meshgrid(_y, _x, indexing="ij")
        y0 = y0[None, None]
        x0 = x0[None, None]

        sign_pad = torch.arange(
            -1,
            1 + 1,
            device=device,
            dtype=torch.float32,
        ).reshape(1, -1, 1, 1)

        if residual_maps is not None:
            residual = residual_maps * sign_pad
            distance_fields = dis_maps + residual
        else:
            distance_fields = dis_maps
        distance_fields = distance_fields.clamp(min=0, max=1.0)
        md_un = (md_maps[:, :1] - 0.5) * np.pi * 2
        st_un = md_maps[:, 1:2] * np.pi / 2.0
        ed_un = -md_maps[:, 2:3] * np.pi / 2.0

        cs_md = md_un.cos()
        ss_md = md_un.sin()

        y_st = torch.tan(st_un)
        y_ed = torch.tan(ed_un)

        x_st_rotated = (cs_md - ss_md * y_st) * distance_fields * scale
        y_st_rotated = (ss_md + cs_md * y_st) * distance_fields * scale

        x_ed_rotated = (cs_md - ss_md * y_ed) * distance_fields * scale
        y_ed_rotated = (ss_md + cs_md * y_ed) * distance_fields * scale

        x_st_final = (x_st_rotated + x0).clamp(min=0, max=width - 1)
        y_st_final = (y_st_rotated + y0).clamp(min=0, max=height - 1)

        x_ed_final = (x_ed_rotated + x0).clamp(min=0, max=width - 1)
        y_ed_final = (y_ed_rotated + y0).clamp(min=0, max=height - 1)

        lines = torch.stack((x_st_final, y_st_final, x_ed_final, y_ed_final), dim=-1)
        if flatten:
            lines = lines.reshape(batch_size, -1, 4)

        return lines

    def compute_loi_features(self, features_per_image, lines_per_im, tspan):

        h, w = features_per_image.size(1), features_per_image.size(2)
        U, V = lines_per_im[:, :2], lines_per_im[:, 2:]

        sampled_points = U[:, :, None] * tspan + V[:, :, None] * (1 - tspan) - 0.5

        sampled_points = sampled_points.permute((0, 2, 1)).reshape(-1, 2)
        px, py = sampled_points[:, 0], sampled_points[:, 1]
        px0 = px.floor().clamp(min=0, max=w - 1)
        py0 = py.floor().clamp(min=0, max=h - 1)
        px1 = (px0 + 1).clamp(min=0, max=w - 1)
        py1 = (py0 + 1).clamp(min=0, max=h - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = (
            features_per_image[:, py0l, px0l] * (py1 - py) * (px1 - px)
            + features_per_image[:, py1l, px0l] * (py - py0) * (px1 - px)
            + features_per_image[:, py0l, px1l] * (py1 - py) * (px - px0)
            + features_per_image[:, py1l, px1l] * (py - py0) * (px - px0)
        )
        xp = (
            xp.reshape(features_per_image.shape[0], -1, tspan.numel())
            .permute(1, 0, 2)
            .contiguous()
        )

        return xp.flatten(1)

    def bilinear_sampling(self, features, points):
        h, w = features.size(1), features.size(2)
        px, py = points[:, 0], points[:, 1]

        px0 = px.floor().clamp(min=0, max=w - 1)
        py0 = py.floor().clamp(min=0, max=h - 1)
        px1 = (px0 + 1).clamp(min=0, max=w - 1)
        py1 = (py0 + 1).clamp(min=0, max=h - 1)
        px0l, py0l, px1l, py1l = px0.long(), py0.long(), px1.long(), py1.long()

        xp = (
            features[:, py0l, px0l] * (py1 - py) * (px1 - px)
            + features[:, py1l, px0l] * (py - py0) * (px1 - px)
            + features[:, py0l, px1l] * (py1 - py) * (px - px0)
            + features[:, py1l, px1l] * (py - py0) * (px - px0)
        )

        return xp

    def get_junctions(self, jloc, joff, topk=300, th=0):
        width = jloc.size(2)
        jloc = jloc.reshape(-1)
        joff = joff.reshape(2, -1)

        scores, index = torch.topk(jloc, k=topk)

        y = (
            torch.div(index, width, rounding_mode="trunc").float()
            + torch.gather(joff[1], 0, index)
            + 0.5
        )
        x = (index % width).float() + torch.gather(joff[0], 0, index) + 0.5

        junctions = torch.stack((x, y)).t()

        if th > 0:
            return junctions[scores > th], scores[scores > th]
        else:
            return junctions, scores

    def forward_lines(self, in_distillation, annotations=None):

        features = in_distillation["features"]
        outputs = torch.cat(
            [in_distillation["afm"], in_distillation["junctions"]], dim=-3
        )

        loi_features = self.fc1(features)
        loi_features_thin = self.fc3(features)
        loi_features_aux = self.fc4(features)

        output = outputs
        md_pred = output[:, :3].sigmoid()
        dis_pred = output[:, 3:4].sigmoid()
        res_pred = output[:, 4:5].sigmoid()
        jloc_pred = output[:, 5:7].softmax(1)[:, 1:]
        joff_pred = output[:, 7:9].sigmoid() - 0.5

        lines_pred = self.hafm_decoding(md_pred, dis_pred, res_pred, flatten=True)[0]

        jloc_pred_nms = non_maximum_suppression(jloc_pred[0])

        topK = min(
            300,
            int((jloc_pred_nms > 0.008).float().sum().item()),
        )

        juncs_pred, _ = self.get_junctions(
            jloc_pred_nms, joff_pred[0], topk=topK, th=0.008
        )

        lines_adjusted, lines_init, perm, _ = self.wireframe_matcher(
            juncs_pred, lines_pred
        )

        e1_features = self.bilinear_sampling(
            loi_features[0], lines_adjusted[:, :2] - 0.5
        ).t()
        e2_features = self.bilinear_sampling(
            loi_features[0], lines_adjusted[:, 2:] - 0.5
        ).t()

        tspan = torch.linspace(0, 1, 32).view(1, 1, 32).to("cuda")

        f1 = self.compute_loi_features(
            loi_features_thin[0], lines_adjusted, tspan=tspan[..., 1:-1]
        )

        f2 = self.compute_loi_features(
            loi_features_aux[0], lines_init, tspan=tspan[..., 1:-1]
        )
        line_features = torch.cat((e1_features, e2_features, f1, f2), dim=-1)
        logits = self.fc2_head(
            self.fc2(line_features) + self.fc2_res(torch.cat((f1, f2), dim=-1))
        )

        scores = logits.softmax(dim=-1)[:, 1]

        sarg = torch.argsort(scores, descending=True)

        lines_final = lines_adjusted[sarg]
        score_final = scores[sarg]

        num_detection = min((score_final > 0.00).sum(), 1000)
        lines_final = lines_final[:num_detection]
        score_final = score_final[:num_detection]

        juncs_final = juncs_pred
        juncs_score = _

        sx = annotations[0]["width"] / output.size(3)
        sy = annotations[0]["height"] / output.size(2)

        lines_final[:, 0] *= sx
        lines_final[:, 1] *= sy
        lines_final[:, 2] *= sx
        lines_final[:, 3] *= sy

        juncs_final[:, 0] *= sx
        juncs_final[:, 1] *= sy

        return {
            "lines_pred": lines_final,
            "lines_score": score_final,
            "juncs_pred": juncs_final,
            "juncs_score": juncs_score,
            "num_proposals": lines_adjusted.size(0),
            "filename": annotations[0]["filename"],
            "width": annotations[0]["width"],
            "height": annotations[0]["height"],
        }

    def forward(self, input):
        with torch.no_grad():
            predictions = self.forward_lines(
                input,
                [{"width": self.size[0], "height": self.size[1], "filename": ""}],
            )
        return predictions


def sample_line_descriptors(lines_pred: Tensor, features: Tensor) -> Tensor:
    b, d = features.shape[:2]
    n_lines = lines_pred.shape[-3]
    lines_pred2 = lines_pred[None].flatten(-3, -2)
    img_w_h = lines_pred2.new_tensor([features.shape[-1], features.shape[-2]])
    normalized_lines = 2 * lines_pred2 / (img_w_h - 1) - 1
    descriptors = grid_sample(features, normalized_lines, align_corners=True)
    return descriptors.reshape(b, d, n_lines, 2).transpose(1, 2).transpose(2, 3)


class WireframeNet(BaseModel):

    default_conf = {
        "weights": "checkpoints/base_model_v4.pth",
        "size": [640, 640],
    }

    required_data_keys = ["image"]

    def _init(self, config):
        self.size = list(config.size)
        self.encoder = UNET()
        self.decoder_DISK = DiskDecoder()
        self.decoder_HAWP = HawpDecoder()
        self.HAWP_Headers = HawpHeaders(self.size)
        self.DISK_Headers = DiskHeaders()
        if config.weights is not None:
            checkpoint = torch.load(project_root / config.weights)
            self.load_state_dict(checkpoint)

        self.are_weights_initialized = True

    def transform_batch(self, batch):
        target_size = self.size
        transformation = transforms.Resize(target_size)
        return transformation(batch)

    def _forward(self, input_dict, threshold=0.02):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        input_batch = input_dict["image"]
        input_original_size = [input_batch.shape[-2], input_batch.shape[-1]]

        input_batch = self.transform_batch(input_batch)

        encoded_features = self.encoder(input_batch)
        decoded_features_disk = self.decoder_DISK(encoded_features)
        output_hawp_decoder = self.decoder_HAWP(encoded_features[1])

        with torch.no_grad():
            result_hawp = self.HAWP_Headers.eval()(output_hawp_decoder)
            result_disk = self.DISK_Headers.eval()(
                decoded_features_disk,
                result_hawp["juncs_pred"],
                device=device,
            )

        indices = WireframeGraph.xyxy2indices(
            result_hawp["juncs_pred"], result_hawp["lines_pred"]
        )
        wireframe = WireframeGraph(
            result_hawp["juncs_pred"],
            result_hawp["juncs_score"],
            indices,
            result_hawp["lines_score"],
            result_hawp["width"],
            result_hawp["height"],
        )

        l_segments = wireframe.line_segments(
            threshold=threshold, device=device, to_np=False
        )
        l_segments = (
            l_segments[:, :-1].reshape(len(l_segments), 2, 2).detach().cpu().numpy()
        )

        l_segments[..., 0] *= input_original_size[1] / self.size[-1]
        l_segments[..., 1] *= input_original_size[0] / self.size[-1]

        points = result_disk[0].float()
        points[..., 0] *= input_original_size[1] / self.size[-1]
        points[..., 1] *= input_original_size[0] / self.size[-1]

        return {
            "line_segments": l_segments,
            "points": points,
            "descriptors": result_disk[1],
        }

    def loss(self, pred, data):  # noqa
        raise NotImplementedError
