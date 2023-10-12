'''generate embeddings from sequence'''
import os
import argparse
import traceback

import torch.distributed as dist
import torch.multiprocessing as mp

from embedders import create_parser, prepare_dataframe, validate_args
from embedders import main_esm, main_prottrans
from embedders.checkpoint import capture_checkpoint, find_and_load_checkpoint_file

def mp_process(rank_id: int, nproc: int, args: argparse.Namespace):
	# set env variables
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'
	os.environ['LOCAL_RANK'] = str(rank_id)
	os.environ['WORLD_SIZE'] = str(nproc)
	dist.init_process_group("nccl", rank=rank_id, world_size=nproc)
	args, df = validate_args(args, verbose=True)
	df, batch_iter = prepare_dataframe(df, args, rank_id=rank_id)
	try:
		if not args.embedder.startswith('esm'):
			main_prottrans(df, args, batch_iter, rank_id=rank_id)
		else:
			main_esm(df, args, batch_iter, rank_id=rank_id)
	except Exception as e:
		capture_checkpoint(args, exception_msg = e, rank_id=rank_id)
		traceback.print_exc()
	except KeyboardInterrupt:
		capture_checkpoint(args, exception_msg = 'keyboard interrput', rank_id=rank_id)
		traceback.print_exc()
    

if __name__ == "__main__":
	args = create_parser()
	if args.subparser_name == 'resume':
		args_t = find_and_load_checkpoint_file(args.output)
		nproc = getattr(args_t, 'nproc', 1) # legacy support
	else:
		nproc = args.nproc
	if nproc == 1:
		args, df = validate_args(args, verbose=True)
		df, batch_iter = prepare_dataframe(df, args)
		try:
			if not args.embedder.startswith('esm'):
				main_prottrans(df, args, batch_iter)
			else:
				main_esm(df, args, batch_iter)
		# checkpointing
		except Exception as e:
			capture_checkpoint(args, exception_msg = e)
			traceback.print_exc()
		except KeyboardInterrupt:
			capture_checkpoint(args, exception_msg = 'keyboard interrput')
			traceback.print_exc()
	else:
		# https://discuss.pytorch.org/t/how-do-i-run-inference-in-parallel/126757/2
		mp.spawn(mp_process,
        args=(nproc, args, ),
        nprocs=nproc,
        join=True)