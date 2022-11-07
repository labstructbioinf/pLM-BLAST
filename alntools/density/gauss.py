import torch as th

PI = th.FloatTensor([3.14159])


def gaussian_function(theta, X, Y, sigma_X, sigma_Y):
    '''
    calc gaussian density (unnormalized)
    '''
    # https://en.wikipedia.org/wiki/Gaussian_function
    x0 = X.shape[0]/2
    y0 = Y.shape[0]/2
    theta = th.FloatTensor([theta])

    a = ((th.cos(theta)**2) / (2 * sigma_X**2)) + (th.sin(theta)**2 / (2 * sigma_Y**2))
    b = -th.sin(2 * theta) / (4 * sigma_X**2) + th.sin(2 * theta) / (4 * sigma_Y**2)
    c = ((th.sin(theta)**2) / (2 * sigma_X**2)) + (th.cos(theta)**2 / (2 * sigma_Y**2))
    
    Z =  th.exp(-(a * (X - x0)**2 + 2 * b * (X - x0) * (Y - y0) + c * (Y - y0)**2));
    return Z


def gaussian_from_shape(xdim, ydim):
    '''
    create gaussian kernel from given shapes
    '''
    factor = xdim/ydim
    theta = factor*PI/4
    xrange = th.arange(0, xdim)
    yrange = th.arange(0, ydim)
    X, Y = th.meshgrid(xrange, yrange)
    sigma_X, sigma_Y = xdim/4, ydim/2
    kernel = gaussian_function(theta, X, Y, sigma_X, sigma_Y)
    kernel = kernel.view(1,1, xdim, ydim)/kernel.sum()
    return kernel


def diagonal_from_shape(xdim, ydim, device='cpu'):
    
    assert isinstance(xdim, (int, th.long))
    assert isinstance(ydim, (int, th.long))
    
    filt = th.zeros((xdim, ydim), dtype=th.float, device=device)
    filt.fill_diagonal_(1)
    filt.unsqueeze_(0)
    filt.unsqueeze_(0)
    return filt

@th.jit.script
def gaussian_mean(x : th.Tensor, kernel : th.Tensor, stride : int = 1):
    '''
    average image by sliding window with gausian filter
    params:
        x (torch.FloatTensor)
        kernel (torch.FloatTensor)
        stride (int)
    return:
        x_blurred (torch.FloatTensor)
    '''
    x_len = x.size(0)
    if x_len < kernel.shape[0]:
        return th.tensor([0]) 
    x = x.unsqueeze(0).unsqueeze(0)
    #filter /= filter_size**2
    xprim = th.nn.functional.conv2d(x, kernel, stride=(stride,stride))
    return xprim.squeeze()

def measure_local_similarity(density: th.Tensor, num_residues: int):
    '''
    params:
        density map
        num_residues (int) size of desired similarity area
    return:
        max similarity in density map
    '''
    gkernel = gaussian_from_shape(num_residues, num_residues)
    density_blurred = gaussian_mean(density, gkernel)
    return density_blurred.max().item()