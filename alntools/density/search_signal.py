
import torch as th



def smooth_signal(x, filter_size=11):
    '''
    smooth signal with conv filter
    '''
    assert filter_size % 2 != 0
    if not isinstance(x, th.FloatTensor):
        x = x.float()
    xmax = x.max()
    x = x.unsqueeze(0).unsqueeze(0)
    filter = th.ones((1,1,filter_size))
    xprim = th.nn.functional.conv1d(x, filter, padding=(filter_size-1)//2)
    xprim_max = xprim.max()
    xprim = xmax*(xprim/xprim_max)
    
    return xprim.squeeze()

def smooth_image(x, filter_size=11, stride=2):
    '''
    average image by sliding window
    '''
    x_len = x.size(0)
    if x_len < filter_size:
        return th.tensor([0])
    
    x = x.unsqueeze(0).unsqueeze(0)
    filter = th.ones((1,1,filter_size, filter_size))
    #filter /= filter_size**2
    xprim = th.nn.functional.conv2d(x, filter, stride=(stride,stride))
    
    return xprim.squeeze()

def peak_width(x, loc):
    '''
    find peak width
    return:
        start, stop (int) indices
    '''
    loc_local = loc.clone()
    adj_max = x[loc_local-10:loc_local+10].argmax()
    loc_local += (adj_max-10)
    before = diff(x[:loc_local]) < 0
    after = diff(x[loc_local:]) > 0

    before = th.nonzero(before)[-1].item()+1
    after = th.nonzero(after)[0].item()+1

    before = int(before)
    after = int(loc_local + after)
    #mask = th.arange(before, after,1, dtype=th.LongTensor)
    return before, after

def diff(x):
    '''
    torch version of numpy np.diff
    '''
    x = x[1:] - x[:-1]
    return x