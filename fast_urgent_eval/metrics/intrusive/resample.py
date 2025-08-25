import math

import numpy as np
import torch
import torch.nn.functional as F
from pystoi.utils import resample_oct, _resample_window_oct
# A scipy.signal dependency is required for filter design with windows.
from scipy.signal import firwin


def resample_poly_pytorch(
        x: torch.Tensor,
        up: int,
        down: int,
        axis: int = -1,
        window=('kaiser', 5.0),
) -> torch.Tensor:
    """
    Resamples a tensor `x` along a given axis using polyphase filtering in PyTorch.

    The function upsamples the signal by `up`, applies a zero-phase low-pass FIR filter,
    and then downsamples by `down`. The output sample rate is `up / down` times the input.

    This implementation is a PyTorch-based equivalent of `scipy.signal.resample_poly`.

    Parameters
    ----------
    x : torch.Tensor
        The data to be resampled.
    up : int
        The upsampling factor.
    down : int
        The downsampling factor.
    axis : int, optional
        The axis of `x` that is resampled. Defaults to -1.
    window : string, tuple, or torch.Tensor, optional
        The window to use for designing the FIR filter, or the filter coefficients
        themselves. See `scipy.signal.firwin` for details. Defaults to ('kaiser', 5.0).

    Returns
    -------
    torch.Tensor
        The resampled tensor.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("Input 'x' must be a torch.Tensor.")
    if up <= 0 or down <= 0 or up != int(up) or down != int(down):
        raise ValueError("upsampling and downsampling factors must be positive integers.")

    # Simplify the up/down factors
    g = math.gcd(up, down)
    up //= g
    down //= g

    if up == 1 and down == 1:
        return x.clone()

    # --- 1. FIR Filter Design ---
    if isinstance(window, torch.Tensor):
        h = window.to(x)
        if h.ndim != 1:
            raise ValueError("Filter window must be a 1D tensor.")
        half_len = (h.shape[0] - 1) // 2
    else:
        raise ValueError("Only torch.Tensor window is supported in this implementation.")

    # Scale filter by upsampling factor
    h = h * up

    # --- 2. Prepare Tensor for 1D Convolution ---
    # `conv1d` expects (N, C_in, L_in). We move the target axis to the end
    # and treat all other dimensions as the batch dimension.
    if axis != -1 and axis != x.ndim - 1:
        x = x.transpose(axis, -1)

    original_shape = x.shape
    # Flatten all dimensions except the last one
    x_flat = x.reshape(-1, 1, original_shape[-1])
    n_in = original_shape[-1]

    # --- 3. Polyphase Resampling using Convolution ---

    # This logic matches the alignment and padding of scipy's implementation
    # to achieve a zero-phase filtering effect.
    n_pre_pad = (down - half_len % down) % down
    n_post_pad = 0  # Typically zero, used as a safeguard in scipy
    h_padded = F.pad(h, (n_pre_pad, n_post_pad))
    filt_len = h_padded.shape[0]

    # The 'full' convolution output length is L_in + L_filt - 1.
    # We implement this with padding in conv1d.
    # The filter needs to be flipped for convolution to act as a filter.
    h_flipped = torch.flip(h_padded, dims=[-1]).view(1, 1, -1)

    # Upsample by inserting zeros
    # This can be done efficiently by using a strided transposed convolution,
    # but a direct approach is clearer and often fast enough.
    x_up = torch.zeros((x_flat.shape[0], 1, n_in * up), dtype=x.dtype, device=x.device)
    x_up[..., ::up] = x_flat

    # Convolve and then downsample
    y = F.conv1d(x_up, h_flipped, padding=filt_len - 1)
    y_down = y[..., ::down]

    # --- 4. Crop Output to Match SciPy's Alignment and Length ---
    n_out = (n_in * up) // down + (1 if (n_in * up) % down else 0)
    n_pre_remove = (half_len + n_pre_pad) // down

    y_cropped = y_down[..., n_pre_remove: n_pre_remove + n_out]

    # --- 5. Reshape and Return ---
    # Reshape back to original batch dimensions
    final_shape = list(original_shape)
    final_shape[-1] = y_cropped.shape[-1]
    y_final = y_cropped.reshape(final_shape)

    # Transpose axis back to its original position
    if axis != -1 and axis != x.ndim - 1:
        y_final = y_final.transpose(axis, -1)

    return y_final


class ResampleOctavePyTorch(torch.nn.Module):
    """
    A PyTorch module for resampling tensors by a rational factor `p/q`
    using a filter designed to match Octave's resampling method.

    This module upsamples the signal by `p`, applies a zero-phase low-pass FIR filter,
    and then downsamples by `q`. The output sample rate is `p / q` times the input.

    Parameters
    ----------
    new : int
        The new sampling frequency (p).
    dtype : torch.dtype, optional
        The desired data type for filter computation. Defaults to torch.float64.
    """

    def __init__(
        self,
        new: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        super(ResampleOctavePyTorch, self).__init__()
        self.p = new
        self.dtype = dtype

        self.valid_orig = [8000, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000]

        for fs in self.valid_orig:
            h = _resample_window_oct(self.p, fs)
            window = h / np.sum(h)
            window = torch.tensor(window, dtype=dtype)
            self.register_buffer(f'filter_{fs}', window, persistent=False)


    def forward(self, x: torch.Tensor, orig_fs: int) -> torch.Tensor:
        """
        Resamples the input tensor `x`.

        Parameters
        ----------
        x : torch.Tensor
            The data to be resampled.
        orig_fs : int
            The original sampling frequency of the input tensor.

        Returns
        -------
        torch.Tensor
            The resampled tensor.
        """
        return resample_poly_pytorch(
            x, self.p, orig_fs, axis=-1,
            window=getattr(self, f'filter_{orig_fs}'),
        )


if __name__ == "__main__":

    x = torch.randn(16000).double()
    y = torch.randn(16000).double()

    x_resampled_oct = resample_oct(x.numpy(), 10000, 16000)
    y_resampled_oct = resample_oct(y.numpy(), 10000, 16000)\

    resample_fn = ResampleOctavePyTorch(new=10000, dtype=torch.float32)

    x_resampled_torch = resample_fn(x, 16000)
    y_resampled_torch = resample_fn(y, 16000)

    print(x_resampled_oct.shape, x_resampled_torch.shape)
    print(y_resampled_oct.shape, y_resampled_torch.shape)

    print("Max abs diff (x): ", torch.max(torch.abs(torch.tensor(x_resampled_oct) - x_resampled_torch)).item())
    print("Max abs diff (y): ", torch.max(torch.abs(torch.tensor(y_resampled_oct) - y_resampled_torch)).item())