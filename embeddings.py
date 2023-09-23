'''generate embeddings from sequence'''
import re
import os
import tempfile
import warnings

from embedders import create_parser, prepare_dataframe, validate_args
from embedders import main_esm, main_prottrans, main_prost
from embedders.checkpoint import capture_checkpoint


if __name__ == "__main__":
	args = create_parser()
	df = validate_args(args, verbose=True)
	df, num_batches = prepare_dataframe(df, args.batch_size, args.truncate)
	try:
		if args.embedder == 'pt':
			main_prottrans(df, args, num_batches)
		elif args.embedder.startswith('esm'):
			main_esm(df, args, num_batches)
		elif args.embedder.startswith('prost'):
			main_prost(df, args, num_batches)
	except Exception as e:
		# checkpoint calculations
		if capture_checkpoint(args, exception_msg = e):
			print('checkpoint captured')