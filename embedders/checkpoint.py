import argparse
import json
import os


BASE_CHECKPOINT_NAME: str = 'emb_checkpoint.json'
BASE_CHECKPOINT_NAME_MP: str = 'emb_checkpoint_mp_{rank}.json'

def get_checkpoint_name_for_process() -> str:
    '''
    handle correct checkpoint name for multiprocessing
    '''
    if os.environ.get('WORLD_SIZE') is not None:
        fname_checkpoint = BASE_CHECKPOINT_NAME_MP.format(rank=os.environ.get('LOCAL_RANK'))
    else:
        fname_checkpoint = BASE_CHECKPOINT_NAME
    return fname_checkpoint


def checkpoint_from_json(checkpoint_file: os.PathLike) -> argparse.Namespace:
    '''
    read checkpoint file and convert it to namespace
    '''
    fname = get_checkpoint_name_for_process()
    # if directory is given search for checkpoint_file
    if os.path.isdir(checkpoint_file):
        # search for checkpoint of single process
        checkpoint_file = os.path.join(checkpoint_file, fname)
    # h5py case
    if os.path.isfile(checkpoint_file):
        checkpoint_file = checkpoint_file + '_' + fname
        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(f'no checkpoint file in path: {checkpoint_file}')    
    with open(checkpoint_file, 'rt') as fp:
        checkpoint_data = json.load(fp)
    # create empty namepsace
    temp_args = argparse.Namespace()
    # fill it with json config
    temp_args.__dict__.update(checkpoint_data)
    return temp_args


def capture_checkpoint(args: argparse.Namespace, exception_msg: str):
    '''
    save arguments with current batch index
    '''
    print('capturing checkpoint')
    fname = get_checkpoint_name_for_process()
    if args.asdir:
        outfile = os.path.join(args.output, fname)
    elif args.h5py:
        outfile = args.output + '_' + fname
    args_json = vars(args)
    if exception_msg is not None:
        args_json['exception_message'] = str(exception_msg)
    with open(outfile, 'wt') as fp:
        json.dump(args_json, fp)
    print(f'checkpoint captured in {outfile}')
    