import argparse
import json
import os

CHECKPOINT_DIR: str = "{basename}/{checkpoint}"
CHECKPOINT_FILE: str = "{basename}_{checkpoint}"
BASE_CHECKPOINT_NAME: str = 'emb_checkpoint.json'
BASE_CHECKPOINT_NAME_MP: str = 'emb_checkpoint_mp_{rank}.json'


def get_checkpoint_name_for_process() -> str:
    '''
    handle correct checkpoint name including multiprocessing case
    '''
    if os.environ.get('WORLD_SIZE') is not None:
        if 'LOCAL_RANK' not in os.environ:
            raise KeyError('LOCAL_RANK env variable is not set')
        fname_checkpoint = BASE_CHECKPOINT_NAME_MP.format(rank=os.environ.get('LOCAL_RANK'))
    else:
        fname_checkpoint = BASE_CHECKPOINT_NAME
    print('loading checkpoint: ', fname_checkpoint)
    return fname_checkpoint


def find_and_load_checkpoint_file(output: str) -> argparse.Namespace:
    '''
    look for checkpoint file, in single and multiprocess case handle:
    * `output` is checkpoint file
    * `output` is directory with checkpoint file
    * `output` is an embedder output and checkpoint file is substring of `output`
    Returns:
        args: (arparse.Namespace)
    '''
    assert isinstance(output, str)

    checkpoint_correct: str = ""
    checkpoint_mp = BASE_CHECKPOINT_NAME_MP.format(rank='0')
    # output embedding file is given
    if os.path.isfile(output) and not output.endswith('.json'):
        output_dir = os.path.dirname(output)
        filelist = [os.path.join(output_dir, file) for file in os.listdir(output_dir)]
        checkpoint_mp_file = CHECKPOINT_FILE.format(basename=output, checkpoint=checkpoint_mp)
        checkpoint_file = CHECKPOINT_FILE.format(basename=output, checkpoint=BASE_CHECKPOINT_NAME)
        if (checkpoint_mp_file in filelist) and (checkpoint_file in filelist):
            raise FileExistsError(f"""
                path: {output} contain both single process checkpoint and mp version
                remove {checkpoint_file} to proceed in multiprocess or 
                {checkpoint_mp_file} to proceed in single process
                                         """)
        elif checkpoint_mp_file in filelist:
            checkpoint_correct = checkpoint_mp_file
        elif checkpoint_file in filelist:
            checkpoint_correct = checkpoint_file
        else:
            raise FileNotFoundError(f"no checkpoint file for given file: {output}")
    # directory case
    elif os.path.isdir(output):
        filelist = os.listdir(output)
        checkpoint_mp_file = CHECKPOINT_DIR.format(basename=output, checkpoint=checkpoint_mp)
        checkpoint_file = CHECKPOINT_DIR.format(basename=output, checkpoint=BASE_CHECKPOINT_NAME)
        if checkpoint_mp_file in filelist and checkpoint_file in filelist:
            raise FileExistsError(f"""
                path: {output} contain both single process checkpoint to resume calculations
                remove {checkpoint_file} to proceed in multiprocess or 
                {checkpoint_mp_file} to proceed in single process
                                         """)
        elif checkpoint_mp_file in filelist:
            checkpoint_correct = checkpoint_mp_file
        elif checkpoint_file in filelist:
            checkpoint_correct = checkpoint_file
        else:
            raise FileNotFoundError(f"no checkpoint file for given file: {output}")
    # checkpoint file given directly
    elif os.path.isfile(output) and output.endswith('.json'):
        checkpoint_correct = output
    else:
        raise FileNotFoundError(f"no checkpoint file for given file: {output}")
    args = dict_to_namespace(checkpoint_correct)
    return args


def dict_to_namespace(file: str) -> argparse.Namespace:
    '''
    read json and convert it to namespace
    '''
    with open(file, 'rt') as fp:
        checkpoint_data = json.load(fp)
    # create empty namepsace
    temp_args = argparse.Namespace()
    # fill it with json config
    temp_args.__dict__.update(checkpoint_data)
    return temp_args


def checkpoint_from_json(checkpoint_file: os.PathLike) -> argparse.Namespace:
    '''
    read checkpoint file and convert it to namespace
    '''
    fname = get_checkpoint_name_for_process()
    # if directory is given search for checkpoint_file
    if os.path.isdir(checkpoint_file):
        # search for checkpoint of single process
        checkpoint_file = CHECKPOINT_DIR.format(basename=checkpoint_file, checkpoint=fname)
    # h5py case
    if os.path.isfile(checkpoint_file) and not checkpoint_file.endswith('.json'):
        # add suffix
        checkpoint_file = CHECKPOINT_FILE.format(basename=checkpoint_file, checkpoint=fname)
        if not os.path.isfile(checkpoint_file):
            raise FileNotFoundError(f'no checkpoint file in path: {checkpoint_file}')
    if os.path.isfile(checkpoint_file) and checkpoint_file.endswith('.json'):
        checkpoint_file = os.path.join(os.path.dirname(checkpoint_file), fname)
    temp_args = dict_to_namespace(checkpoint_file)
    return temp_args


def capture_checkpoint(args: argparse.Namespace, exception_msg: str, rank_id: int = None):
    '''
    save arguments with current batch index
    '''
    print('capturing checkpoint')
    fname = get_checkpoint_name_for_process()
    if args.asdir:
        outfile = CHECKPOINT_DIR.format(basename=args.output, checkpoint=fname)
    elif args.h5py:
        outfile = CHECKPOINT_FILE.format(basename=args.output, checkpoint=fname)
    args_json = vars(args)
    if exception_msg is not None:
        args_json['exception_message'] = str(exception_msg)
    # add process id for mutliprocess case
    args_json['rank_id'] = rank_id
    with open(outfile, 'wt') as fp:
        json.dump(args_json, fp)
    print(f'checkpoint captured: {outfile}')
    