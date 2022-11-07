import torch as th
diff_tensor = lambda tens: tens[1:] - tens[:-1]

def reverse_index(idx, kernel_size):
    '''
    returns index before conv average
    '''
    assert idx > -1
    
    adj = 0
    if kernel_size % 2 == 0:
        adj = 1
        
    idx_before = (idx - adj)*2 + kernel_size//2
    
    return idx_before


def region_indices(idx, kernel_size):
    
    start_idx = idx - kernel_size//2 - 1
    stop_idx = idx + kernel_size//2 + 1
    
    return th.arange(start_idx, stop_idx)



def span_to_matrix(span, diag_val):
    '''
    refactor span indices to 2d matrix indices
    params:
        span boundaries of contuinity span
        diag_val coords of off diagonal
    '''
    if span.ndim > 1:
        span = span.squeeze()
    arange = th.arange(*span)
    matrix_indices = diag_val + arange.unsqueeze(1)
    return matrix_indices

def check_colinearity(samples, reference):
    '''
    check if any os spans from `samples` matches `reference` span
    params:
        samples (torch.LongTensor 2d shape: (X, 2))
        reference (torch.LongTensor 2d shape: (X, 2))
    return:
        matched samples None if tensors didnt match
    
    '''
    assert samples.ndim > 1
    if reference.ndim > 1:
        # start stop has shape of (num, 1)
        refx, refy = reference.split(1, 1)
    else:
        refx, refy = reference
        refx, refy = start.unsqueeze(0), stop.unsqueeze(0)
    sx, sy = samples[:, 0], samples[:, 1]
    if samples.shape[0] > 1:
        sx, sy = sx.T, sy.T
    #reverse condition
    #check if points are not aligned
    condition = (sy < refx)  #samples are RHS ref
    condition |= (sx > refy) #LHS
    #condition has shape of num of reference samples, num of sample samples
    #if logically summed (.any() method) over first dimension produces condition for each
    # of `samples` matched any of `reference` samples
    condition = (~condition).any(0)
    
    if condition.any():
        return samples[condition]
    
    
    
def refactor_2d_mask_to_diagonals(arr):
    
    if arr.ndim > 2:
        image = arr.squeeze()
    else:
        image = arr.clone()
    num_sl = image.shape[-2]
    num_sr = image.shape[-1]
    shifts = dict()
    for shift in range(-num_sl, num_sr):
        line_bool = image.diagonal(shift) # get diagonal
        line_con_mask = line_bool & line_bool.roll(1) #continuity condition
        #line_con_mask = line_bool
        if ~line_con_mask.any(): # is tensor not empty - no regions
            continue
        line_con = th.nonzero(line_con_mask)
        #print('regions:', line_con)
        line_con_diff = diff_tensor(line_con) != 1 # if True refer to region boundary
        line_con_diff = line_con_diff.flatten()
        lcd_mask = th.nonzero(line_con_diff).flatten()

        #print('boundaries:',lcd_mask)
        if line_con_diff.all(): #all regions contain 1-2 elements
            continue
        if lcd_mask.shape[0] == 0: # check if all matched indices belong to one region 
            coords = th.tensor([line_con[0], line_con[-1]-1]).unsqueeze(0) #if so 
        else:
            region_stops = th.cat((lcd_mask, th.tensor([line_con.shape[0]-1])))
            region_starts = region_stops.roll(1) + 1 #roll right and append zero at the begining
            region_starts[0] = 0
            starts = line_con[region_starts]
            stops = line_con[region_stops]
            '''
            if (starts < stops).any():
                print('err')
                break
            '''
            coords = th.cat((starts, stops), dim=1)

            '''
            if ((stops - starts) > 100).any():
                print('err')
                break
            '''
        if coords.shape[0] == 0:
            continue
        shifts[shift] = coords
    return shifts
