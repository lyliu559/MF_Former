import torch
import torch.nn.functional as F


TP_MEAN = 0.00010232621
TP_STD = 0.0004017048


def inverse_transform(data, mean=TP_MEAN, std=TP_STD):
    """Inverse normalization for total precipitation."""
    return data * std + mean


def _get_reduce_dims(x):
    if x.ndim == 5:      # [seq_len, batch_size, channels, height, width]
        return [2, 3, 4]
    elif x.ndim == 4:    # [batch_size, seq_len, height, width] or similar
        return [1, 2, 3]
    elif x.ndim == 3:    # [channels, height, width]
        return [0, 1, 2]
    else:
        raise ValueError(f"Unsupported tensor shape: {x.shape}")


def crosstab_evaluate(output, ground_truth, threshold_low, threshold_high):
    """
    Compute POD, FAR, and CSI under a given precipitation threshold range.
    Metrics are computed after inverse normalization.
    """
    output_phys = inverse_transform(output)
    gt_phys = inverse_transform(ground_truth)

    reduce_dims = _get_reduce_dims(output)

    pred_mask = ((output_phys >= threshold_low) & (output_phys <= threshold_high)).int()
    gt_mask = ((gt_phys >= threshold_low) & (gt_phys <= threshold_high)).int()

    no_rain_index = torch.eq(torch.sum(gt_mask, dim=reduce_dims), 0)

    hits = torch.sum(pred_mask * gt_mask, dim=reduce_dims)
    misses = torch.sum(gt_mask * (1 - pred_mask), dim=reduce_dims)
    false_alarms = torch.sum(pred_mask * (1 - gt_mask), dim=reduce_dims)

    pod = hits.float() / (hits + misses).float().clamp_min(1e-12)
    far = false_alarms.float() / (hits + false_alarms).float().clamp_min(1e-12)
    csi = hits.float() / (hits + misses + false_alarms).float().clamp_min(1e-12)

    far = torch.where(torch.isnan(far), torch.full_like(far, 1.0), far)

    pod = torch.where(no_rain_index, torch.full_like(pod, 0.0), pod)
    far = torch.where(no_rain_index, torch.full_like(far, 0.0), far)
    csi = torch.where(no_rain_index, torch.full_like(csi, 0.0), csi)

    return pod, far, csi, no_rain_index.int()


def valid(
    model,
    dataloader,
    eval_by_seq=False,
    return_event_scores=False,
    threshold_low=0.0002,
    threshold_high=1.0,
):
    """
    Validation / test function.

    Parameters
    ----------
    return_event_scores : bool
        False -> return rmse, mae
        True  -> return rmse, mae, pod, far, csi
    """
    model.eval()

    pod_list = []
    far_list = []
    csi_list = []
    index_list = []

    rmse_total = 0.0
    mae_total = 0.0

    device = next(model.parameters()).device

    with torch.no_grad():
        for input_mm, target_mm in dataloader:
            input_mm = input_mm.to(device, non_blocking=True)
            target_mm = target_mm.to(device, non_blocking=True)

            output = model(input_mm)
            target_tp = target_mm[:, :, 0, :, :]

            mse_loss = F.mse_loss(output, target_tp, reduction="mean")
            rmse = torch.sqrt(mse_loss).item()
            mae = F.l1_loss(output, target_tp, reduction="mean").item()

            rmse_total += rmse
            mae_total += mae

            if return_event_scores:
                pod_, far_, csi_, index_ = crosstab_evaluate(
                    output, target_tp, threshold_low, threshold_high
                )
                pod_list.append(pod_)
                far_list.append(far_)
                csi_list.append(csi_)
                index_list.append(index_)

    model.train()

    avg_rmse = rmse_total / len(dataloader)
    avg_mae = mae_total / len(dataloader)

    if not return_event_scores:
        return avg_rmse, avg_mae

    index = torch.cat(index_list, dim=0)

    data_num = index.numel()
    if eval_by_seq:
        cal_num = index.size(1) - torch.sum(index, dim=1)
        cal_num = cal_num.clamp_min(1)
        pod = torch.sum(torch.cat(pod_list, dim=1), dim=1) / cal_num
        far = torch.sum(torch.cat(far_list, dim=1), dim=1) / cal_num
        csi = torch.sum(torch.cat(csi_list, dim=1), dim=1) / cal_num
    else:
        cal_num = (data_num - torch.sum(index)).clamp_min(1)
        pod = torch.sum(torch.cat(pod_list, dim=0)) / cal_num
        far = torch.sum(torch.cat(far_list, dim=0)) / cal_num
        csi = torch.sum(torch.cat(csi_list, dim=0)) / cal_num

    pod = pod.item() if torch.numel(pod) == 1 else pod
    far = far.item() if torch.numel(far) == 1 else far
    csi = csi.item() if torch.numel(csi) == 1 else csi

    return avg_rmse, avg_mae, pod, far, csi