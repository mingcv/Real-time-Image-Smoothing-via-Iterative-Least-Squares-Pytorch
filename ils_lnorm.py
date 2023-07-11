import torch


def diff_h(x):
    return torch.cat([x[..., 1:] - x[..., :-1], x[..., :1] - x[..., -1:]], dim=-1)


def diff_v(x):
    return torch.cat([x[..., 1:, :] - x[..., :-1, :], x[..., :1, :] - x[..., -1:, :]], dim=-2)


def idiff_h(x):
    return torch.cat([x[..., -1:] - x[..., :1], x[..., :-1] - x[..., 1:]], dim=-1)


def idiff_v(x):
    return torch.cat([x[..., -1:, :] - x[..., :1, :], x[..., :-1, :] - x[..., 1:, :]], dim=-2)


def psf2otf(size):
    psf = torch.zeros((2, *size), device='cuda')
    psf[:, 0, 0] = -1
    psf[0, 0, -1] = 1
    psf[1, -1, 0] = 1
    return torch.fft.fft2(psf)


def ILS_LNorm(x, otf, lambd, p, itr):
    eps = 1e-4
    gamma = 0.5 * p - 1
    c = p * eps ** gamma

    dfx, dfy = otf.chunk(2)
    dnorm = torch.abs(dfx) ** 2 + torch.abs(dfy) ** 2
    dnorm = 1 + 0.5 * c * lambd * dnorm.repeat(*x.shape[:2], 1, 1)
    fnorm = torch.fft.fft2(x)
    for k in range(itr):
        uh, uv = diff_h(x), diff_v(x)
        muh = c * uh - p * uh * (uh * uh + eps) ** gamma
        muv = c * uv - p * uv * (uv * uv + eps) ** gamma
        normh, normv = idiff_h(muh), idiff_v(muv)
        fu = (fnorm + 0.5 * lambd * torch.fft.fft2(normh + normv)) / dnorm
        x = torch.fft.ifft2(fu).real
        fnorm = fu
    return x


def smooth_filtering(x, otf=None, lambd=1.0, p=0.8, itr=4):
    # When input sizes are fixed, using a given otf is a much faster way.
    if otf is None:
        otf = psf2otf(x.shape[2:])
    smoothed = ILS_LNorm(x, otf, lambd, p, itr)
    return torch.clip(smoothed, 0, 1)


if __name__ == '__main__':
    import PIL.Image as Image
    import torchvision.transforms.functional as TF

    im = Image.open('./flower.png')
    im = TF.to_tensor(im).unsqueeze(0).cuda()
    out = smooth_filtering(im)
    TF.to_pil_image(out[0]).show()
