'''generate embeddings from sequence'''
import os
import sys
import traceback

import torch.distributed as dist
import torch.multiprocessing as mp

from embedders import create_parser, prepare_dataframe, validate_args
from embedders import main_esm, main_prottrans
from embedders.checkpoint import capture_checkpoint

def mp_process(rank_id, args, df):
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '12355'

	dist.init_process_group("nccl", rank=rank_id, world_size=args.nproc)
	df, batch_iter = prepare_dataframe(df, args, rank_id=rank_id)
	main_prottrans(df, args, batch_iter, rank_id=rank_id)
    

if __name__ == "__main__":
	args = create_parser()
	df = validate_args(args, verbose=True)
	if args.nproc == 1:
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
        args=(args, df, ),
        nprocs=args.nproc,
        join=True)