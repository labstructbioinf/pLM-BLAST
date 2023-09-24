'''generate embeddings from sequence'''
import sys
import traceback

from embedders import create_parser, prepare_dataframe, validate_args
from embedders import main_esm, main_prottrans
from embedders.checkpoint import capture_checkpoint


if __name__ == "__main__":
	args = create_parser()
	df = validate_args(args, verbose=True)
	df, batch_iter = prepare_dataframe(df, args)
	try:
		if not args.embedder.startswith('esm'):
			main_prottrans(df, args, batch_iter)
		else:
			main_esm(df, args, batch_iter)
	except Exception as e:
		# checkpoint calculations
		capture_checkpoint(args, exception_msg = e)
		traceback.print_exc()
	except KeyboardInterrupt:
		capture_checkpoint(args, exception_msg = 'keyboard interrput')