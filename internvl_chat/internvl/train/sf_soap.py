import random
import string
from itertools import chain

import torch
import torch.optim as optim

_mode = None

if _mode is None:
    def decorator(func):
        return func
else:
    decorator = torch.compile(fullgraph=False, dynamic=True, mode=_mode)

_einsum_base = string.ascii_lowercase + string.ascii_uppercase


def _merge_dims(grad, max_precond_dim):
    """
    Merges dimensions of the gradient tensor till the product of the dimensions is less than or equal to max_precond_dim.
    """
    shape = grad.shape
    new_shape = []

    curr_shape = 1
    for sh in shape:
        temp_shape = curr_shape * sh
        if temp_shape >= max_precond_dim:
            if curr_shape > 1:
                new_shape.append(curr_shape)
                curr_shape = sh
            else:
                new_shape.append(sh)
                curr_shape = 1
        else:
            curr_shape = temp_shape

    if curr_shape > 1 or len(new_shape) == 0:
        new_shape.append(curr_shape)

    new_grad = grad.reshape(new_shape)
    return new_grad


def set_(dst: torch.Tensor, src: torch.Tensor):
    if dst.is_contiguous():
        dst.set_(src)
    else:
        dst.copy_(src)


@decorator
def _get_orthogonal_matrix_QR(GG, Q, exp_avg_sq, max_precond_dim=10000, merge_dims=False):
    """
    Computes the eigenbases of the preconditioner using one round of power iteration
    followed by torch.linalg.qr decomposition.
    """
    matrix = []
    orth_matrix = []
    for m, o in zip(GG, Q):
        if len(m) == 0:
            matrix.append([])
            orth_matrix.append([])
            continue
        if m.data.dtype != torch.float:
            matrix.append(m.data.float())
            orth_matrix.append(o.data.float())
        else:
            matrix.append(m.data.float())
            orth_matrix.append(o.data.float())

    orig_shape = exp_avg_sq.shape
    if merge_dims:
        exp_avg_sq_new = _merge_dims(exp_avg_sq, max_precond_dim)
    else:
        exp_avg_sq_new = exp_avg_sq

    for ind, (m, o, q) in enumerate(zip(matrix, orth_matrix, Q)):
        if len(m) == 0:
            continue

        est_eig = o.T @ m @ o
        est_eig = torch.diag(est_eig)
        sort_idx = torch.argsort(est_eig, descending=True)
        del est_eig
        torch.index_select(exp_avg_sq_new.clone(), ind, sort_idx, out=exp_avg_sq_new)
        o = o[:, sort_idx]
        power_iter = m @ o
        del o
        set_(q, torch.linalg.qr(power_iter)[0].contiguous())
        del power_iter

    if exp_avg_sq_new.data_ptr() != exp_avg_sq.data_ptr():
        if merge_dims:
            exp_avg_sq_new = exp_avg_sq_new.reshape(orig_shape)
        set_(exp_avg_sq, exp_avg_sq_new)


def _get_orthogonal_matrix(mat):
    """
    Computes the eigenbases of the preconditioner using torch.linalg.eigh decomposition.
    """
    matrix = []
    for m in mat:
        if len(m) == 0:
            matrix.append([])
            continue
        if m.data.dtype != torch.float:
            float_data = False
            original_type = m.data.dtype
            original_device = m.data.device
            matrix.append(m.data.float())
        else:
            float_data = True
            matrix.append(m.data)

    final = []
    for m in matrix:
        if len(m) == 0:
            final.append([])
            continue
        try:
            _, Q = torch.linalg.eigh(m + 1e-30 * torch.eye(m.shape[0], device=m.device))
        except:
            try:
                _, Q = torch.linalg.eigh(m.to(torch.float64) + 1e-30 * torch.eye(m.shape[0], device=m.device))
                Q = Q.to(m.dtype)
            except torch.OutOfMemoryError:
                _, Q = torch.linalg.eigh(
                    m.to(device='cpu', dtype=torch.float64) + 1e-30 * torch.eye(m.shape[0], device='cpu',
                                                                                dtype=torch.float64))
                Q = Q.to(m.device, m.dtype)
        Q = torch.flip(Q, [1])

        if not float_data:
            Q = Q.to(original_device).type(original_type)
        final.append(Q)
    return final


@decorator
def _compute_ggt(grad, GG, max_precond_dim, merge_dims, precondition_1d, beta):
    if grad.dim() == 1:
        if precondition_1d and grad.shape[0] <= max_precond_dim:
            GG[0].lerp_(grad.unsqueeze(1) * grad.unsqueeze(0), 1 - beta)
        return

    if merge_dims:
        new_grad = _merge_dims(grad, max_precond_dim)
        for idx, sh in enumerate(new_grad.shape):
            if sh <= max_precond_dim:
                outer_product = torch.tensordot(new_grad, new_grad,
                                                dims=[[*chain(range(idx), range(idx + 1, len(new_grad.shape)))]] * 2)
                GG[idx].lerp_(outer_product, 1 - beta)
        return

    for idx, sh in enumerate(grad.shape):
        if sh <= max_precond_dim:
            outer_product = torch.tensordot(grad, grad,  # Contracts across all dimensions except for k.
                                            dims=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2)
            GG[idx].lerp_(outer_product, 1 - beta)


def _update_preconditioner(grad, state, max_precond_dim, merge_dims, precondition_1d, beta, update_precond):
    """
    Updates the preconditioner matrices and the eigenbases (L, R, Q_L, Q_R in the paper).
    """
    _compute_ggt(grad, state['GG'], max_precond_dim, merge_dims, precondition_1d, beta)
    if state['Q'] is None:
        state['Q'] = _get_orthogonal_matrix(state['GG'])
    if update_precond:
        _get_orthogonal_matrix_QR(state['GG'], state['Q'], state['exp_avg_sq'], max_precond_dim, merge_dims)


def _init_preconditioner(grad, state, max_precond_dim=10000, precondition_1d=False, merge_dims=False):
    """
    Initializes the preconditioner matrices (L and R in the paper).
    """
    state['GG'] = []  # Will hold all the preconditioner matrices (L and R in the paper).
    if grad.dim() == 1:
        if not precondition_1d or grad.shape[0] > max_precond_dim:
            state['GG'].append([])
        else:
            state['GG'].append(torch.zeros(grad.shape[0], grad.shape[0], device=grad.device))
    else:
        if merge_dims:
            grad = _merge_dims(grad, max_precond_dim)

        for sh in grad.shape:
            if sh > max_precond_dim:
                state['GG'].append([])
            else:
                state['GG'].append(torch.zeros(sh, sh, device=grad.device))

    state['Q'] = None  # Will hold all the eigenbases of the preconditioner.


@decorator
def _project(grad, Q, merge_dims, max_precond_dim, back: bool):
    """

    :param grad:
    :param Q:
    :param merge_dims:
    :param max_precond_dim:
    :param back: whether to project to Shampoo eigenbases or back to original space
    :return:
    """
    original_shape = grad.shape
    if merge_dims:
        grad = _merge_dims(grad, max_precond_dim)

    param = _einsum_base[:grad.dim()]
    preconditioners = ",".join(
        (g + g.upper())[::-1 if back else 1] for m, g in zip(Q, param) if isinstance(m, torch.Tensor))
    if preconditioners:
        out = ''.join(c.upper() if c.upper() in preconditioners else c for c in param)
        grad = torch.einsum(f'{param},{preconditioners}->{out}', grad, *[q for q in Q if isinstance(q, torch.Tensor)])
    if merge_dims:
        grad = grad.reshape(original_shape)
    return grad


class SFPaLMForeachSOAP(optim.Optimizer):
    def __init__(self, params, lr: float = 3e-3, beta=0.9, beta2_scale: float = 0.8, eps: float = 1e-8,
                 weight_decay: float = 0.01, precondition_frequency: int = 2, max_precond_dim: int = 2048,  #
                 merge_dims: bool = True, precondition_1d: bool = False, normalize_grads: bool = False,
                 data_format: str = "channels_first", correct_bias: bool = True, warmup_steps: int = 1, r=0.0,
                 weight_lr_power=2.0, gradient_clip_val: float = 0.1):
        defaults = {"lr": lr, "beta": beta, "beta2_scale": beta2_scale, "eps": eps, "weight_decay": weight_decay,
                    "precondition_frequency": precondition_frequency, "max_precond_dim": max_precond_dim,
                    "merge_dims": merge_dims, "precondition_1d": precondition_1d, "normalize_grads": normalize_grads,
                    "correct_bias": correct_bias, 'warmup_steps': warmup_steps, 'r': r,
                    'weight_lr_power': weight_lr_power, 'train_mode': True, 'step': -1,
                    'gradient_clip_val': gradient_clip_val}
        super().__init__(params, defaults)
        self._data_format = data_format
        self.rng = random.Random(0x120983109)

    def eval(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta']
            if beta1 > 0 and train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to x
                        p.data.lerp_(end=state['z'], weight=1 - 1 / beta1)
                group['train_mode'] = False

    def train(self):
        for group in self.param_groups:
            train_mode = group['train_mode']
            beta1 = group['beta']
            if beta1 > 0 and not train_mode:
                for p in group['params']:
                    state = self.state[p]
                    if 'z' in state:
                        # Set p.data to y
                        p.data.lerp_(end=state['z'], weight=1 - beta1)
                group['train_mode'] = True

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        if closure is None:
            loss = None
        else:
            loss = closure()

        for group in self.param_groups:
            vals = []
            merge_dims = group['merge_dims']
            max_precond_dim = group['max_precond_dim']
            precondition_1d = group['precondition_1d']

            step = group['step'] = group.get("step", -1) + 1


            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                p.grad = None
                vals.append((p, grad))

            p_list, grad = zip(*vals)

            # adaptive gradient clipping
            if group["gradient_clip_val"] > 0:
                p_norm = torch._foreach_norm(p_list)
                g_norm = torch._foreach_norm(grad)
                torch._foreach_maximum_(p_norm, 1e-3)
                torch._foreach_maximum_(g_norm, group["eps"])
                torch._foreach_div_(p_norm, g_norm)
                torch._foreach_mul_(p_norm, group["gradient_clip_val"])
                torch._foreach_minimum_(p_norm, 1)
                torch._foreach_mul_(grad, p_norm)

            vals = []

            for p, grad in zip(p_list, grad):
                state = self.state[p]

                if "z" not in state:
                    state["z"] = torch.clone(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(grad)
                    _init_preconditioner(grad, state, max_precond_dim, precondition_1d, merge_dims)
                    _update_preconditioner(grad, state, max_precond_dim, merge_dims, precondition_1d, 0, True)
                    continue  # first step is skipped so that we never use the current gradients in the projection.

                # Projecting gradients to the eigenbases of Shampoo's preconditioner
                # i.e. projecting to the eigenbases of matrices in state['GG']
                grad_projected = _project(grad, state['Q'], merge_dims, max_precond_dim, False)
                z, exp_avg_sq = state["z"], state["exp_avg_sq"]
                vals.append((p, grad, grad_projected, z, exp_avg_sq))

            if not vals:
                continue

            p_list, grad, grad_projected, z, exp_avg_sq = zip(*vals)
            beta1 = group["beta"]

            beta2 = 1 - max(step, 1) ** -group['beta2_scale']
            bias_correction2 = 1.0 - beta2 ** step
            new_debiased2 = (1 - beta2) / bias_correction2

            # Decay the first and second moment running average coefficient
            # In-place operations to update the averages at the same time
            torch._foreach_mul_(exp_avg_sq, 1 - new_debiased2)
            torch._foreach_addcmul_(exp_avg_sq, grad_projected, grad_projected, value=new_debiased2)
            denom = torch._foreach_sqrt(exp_avg_sq)
            torch._foreach_maximum_(denom, group["eps"])
            torch._foreach_div_(grad_projected, denom)

            update_precond = group['step'] > 0 and group['step'] % group['precondition_frequency'] == 0

            for p, g, gp in zip(p_list, grad, grad_projected):
                state = self.state[p]
                # Projecting back the preconditioned (by Adam) exponential moving average of gradients
                # to the original space
                # CANT DO /= HERE AS EXP_AVG MAY POINT TO THE BUFFER
                set_(gp, _project(gp, state['Q'], merge_dims, max_precond_dim, back=True))

                _update_preconditioner(g, state, max_precond_dim, merge_dims, precondition_1d, 1 - new_debiased2,
                                       update_precond)

            # Weight decay calculated at y
            if group["weight_decay"] > 0:
                torch._foreach_add_(grad, p_list, alpha=group["weight_decay"])

            lr = group["lr"] * min(step / group['warmup_steps'], 1)
            weight = lr ** group['weight_lr_power']
            weight_sum = group['weight_sum'] = group.get('weight_sum', 0) + weight

            try:
                ckp1 = weight / weight_sum
            except ZeroDivisionError:
                ckp1 = 0

            # These operations update y in-place,
            # without computing x explicitly.
            torch._foreach_lerp_(p_list, z, weight=ckp1)
            torch._foreach_add_(p_list, grad_projected, alpha=lr * (beta1 * (1 - ckp1) - 1))

            # z step
            torch._foreach_sub_(z, grad_projected, alpha=lr)
        return loss
