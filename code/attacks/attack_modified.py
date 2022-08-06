import torch

from attacks.attack import Attack
from Datasets.tartanTrajFlowDataset import extract_traj_data


class AttackModified(Attack):
    def __init__(self, model, criterion, test_criterion, norm, data_shape, sample_window_size=None,
                 sample_window_stride=None, pert_padding=(0, 0), optimizer=None):

        super().__init__(model=model,
                         criterion=criterion,
                         test_criterion=test_criterion,
                         norm=norm,
                         data_shape=data_shape,
                         sample_window_size=None,
                         sample_window_stride=None,
                         pert_padding=(0, 0))
        self.optimizer = optimizer

    def gradient_ascent_step(self, pert, data_shape, data_loader, y_list, clean_flow_list,
                             multiplier, a_abs, eps, device=None):
        pert_expand = pert.expand(data_shape[0], -1, -1, -1).to(device)
        grad_tot = torch.zeros_like(pert, requires_grad=False)

        # calculates total gradient
        for data_idx, data in enumerate(data_loader):
            dataset_idx, dataset_name, traj_name, traj_len, \
            img1_I0, img2_I0, intrinsic_I0, \
            img1_I1, img2_I1, intrinsic_I1, \
            img1_delta, img2_delta, \
            motions_gt, scale, pose_quat_gt, patch_pose, mask, perspective = extract_traj_data(data)
            mask1, mask2, perspective1, perspective2 = self.prep_data(mask, perspective)
            grad = self.calc_sample_grad(pert_expand, img1_I0, img2_I0, intrinsic_I0,
                                         img1_delta, img2_delta,
                                         scale, y_list[data_idx], clean_flow_list[data_idx], patch_pose,
                                         perspective1, perspective2,
                                         mask1, mask2, device=device)
            grad = grad.sum(dim=0, keepdims=True).detach()

            # accumulate gradient
            with torch.no_grad():
                grad_tot += grad

            # region delete params
            del grad
            del img1_I0
            del img2_I0
            del intrinsic_I0
            del img1_I1
            del img2_I1
            del intrinsic_I1
            del img1_delta
            del img2_delta
            del motions_gt
            del scale
            del pose_quat_gt
            del patch_pose
            del mask
            del perspective
            torch.cuda.empty_cache()
            # endregion

        # do gradient step
        with torch.no_grad():
            grad = self.normalize_grad(grad_tot)
            pert = self.optimization_update(a_abs, grad, multiplier, pert)
            pert = self.project(pert, eps)

        return pert

    def optimization_update(self, a_abs, grad, multiplier, pert):
        pert += multiplier * a_abs * grad
        return pert