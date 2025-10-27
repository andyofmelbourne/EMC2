import numpy as np

def make_mask(det, model, xyz_offset=[[0, 0, 0]], scale=1, padding=[0, 0]):
    """
    The most important thing is that unmasked pixels fall within the
    model domain for all tomograms:
        0 <= N_sri < model_length
    otherwise we have to keep checking if a slice through the model
    produces valid values and that the inverse mapping (pixel to model)
    falls within the domain. So the model should be 'bigger' than the mask.

    qmax = min((qmax_model-offset_q)/scale)

    model_q = scale R . (pixel_q - offset_q)

    so the biggest model_q value will be:
        model_q = scale_max (pixel_q_max + |offset_q|)

    so pixel_q_max should be no bigger than:
        pixel_q_max = model_q / scale_max - |offset_q|

    we also offset limits for likelihood calculations
    """
    # find maximum offset in q-units
    o = np.array(xyz_offset)
    imax = np.max(np.sum(o[:, :2]**2, axis=1)**0.5 / det.pixel_size)
    qmax_offset = imax * det.dq
    scale_max = np.max(scale)
    qmax = model.qmax / scale_max - qmax_offset
    qmax -= model.dq  # safety margin
    qmin = det.qmin

    qmin += model.dq * padding[0]
    qmax -= model.dq * padding[1]

    mask = det.mask * (det.qr >= qmin) * (det.qr < qmax)
    return mask
