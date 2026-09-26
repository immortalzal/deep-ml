def conv_out_shape(h, w, kernel, stride, padding):
    height = (h + 2 * padding - kernel + 1) // stride
    weight = (w + 2 * padding - kernel + 1) // stride
    return (height, weight)
    """Return (H_out, W_out) for a 2D conv with the given spatial params.

    Args:
        h: input height
        w: input width
        kernel: kernel size (same for H and W)
        stride: stride (same for H and W)
        padding: padding (same for H and W)

    Returns:
        Tuple of ints (H_out, W_out).
    """
    # TODO: apply the standard conv output-size formula
    pass
