import argparse
import json
import os


BASE_CHECKPOINT_NAME: str = 'emb_checkpoint.json'

def checkpoint_from_json(checkpoint_file: os.PathLike) -> argparse.Namespace:
    '''
    read checkpoint file and convert it to namespace
    '''
    # if directory is given search for checkpoint_file
    if os.path.isdir(checkpoint_file):
        # search for checkpoint
        checkpoint_file = os.path.join(checkpoint_file, BASE_CHECKPOINT_NAME)
    # check if file exists
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
    if not args.asdir:
        return None
    print('capturing checkpoint')
    outdir = args.output
    outfile = os.path.join(outdir, BASE_CHECKPOINT_NAME)
    args_json = vars(args)
    if exception_msg is not None:
        args_json['exception_message'] = str(exception_msg)
    with open(outfile, 'wt') as fp:
        json.dump(args_json, fp)
    print(f'checkpoint captured in {outfile}')
    