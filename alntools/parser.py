import os
import argparse

def range_limited_float_type(arg, MIN, MAX):
	""" Type function for argparse - a float within some predefined bounds """
	try:
		f = float(arg)
	except ValueError:
		raise argparse.ArgumentTypeError("Must be a floating point number")
	if f <= MIN or f >= MAX :
		raise argparse.ArgumentTypeError("Argument must be <= " + str(MAX) + " and >= " + str(MIN))
	return f


def get_parser() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description =  
		"""
		Searches a database of embeddings with a query embedding
		""",
		formatter_class=argparse.RawDescriptionHelpFormatter
		)

	range01 = lambda f:range_limited_float_type(f, 0, 1)
	range0100 = lambda f:range_limited_float_type(f, 0, 101)

	# input and output

	parser.add_argument('db', help='database embeddings and index',
						type=str)	

	parser.add_argument('query', help='query embedding and index',
						type=str)	

	parser.add_argument('output', help='''if you want the results to be in separate files,
					  enter the directory path and --mqmf 
					 else type output file and --mqsf''',
						type=str)	

	parser.add_argument('--mqsf', help='Multi query single file', 
			 			action='store_true', default=False)
	
	parser.add_argument('--mqmf', help='Multi query multi file', 
			 			action='store_true', default=False)

	parser.add_argument('--raw', help='skip postprocessing steps and return pickled pandas dataframe with all alignments', 
			 			action='store_true', default=False)
	
	
	# cosine similarity scan

	parser.add_argument('-cosine_percentile_cutoff', help='percentile cutoff for cosine similarity (default: %(default)s). The lower the value, the more sequences will be returned by the pre-screening procedure and aligned with the more accurate but slower pLM-BLAST',
						type=range0100, default=95, dest='COS_PER_CUT')	

	parser.add_argument('-use_chunks', help='use fast chunk cosine similarity screening instead of regular cosine similarity screening. (default: %(default)s)',
			 action='store_true', default=True)

	# plmblast

	parser.add_argument('-alignment_cutoff', help='pLM-BLAST alignment score cut-off (default: %(default)s)',
						type=range01, default=0.3, dest='ALN_CUT')						

	parser.add_argument('-win', help='Window length (default: %(default)s)',
						type=int, default=10, choices=range(50), metavar="[1-50]", dest='WINDOW_SIZE')	

	parser.add_argument('-span', help='Minimal alignment length (default: %(default)s). Must be greater than or equal to the window length',
						type=int, default=25, choices=range(50), metavar="[1-50]", dest='MIN_SPAN_LEN')

	parser.add_argument('--global_aln', help='use global pLM-BLAST alignment. Use only if you expect the query to be a single-domain sequence (default: %(default)s)',
                    	default='False', choices=['True', 'False'])

	parser.add_argument('-gap_ext', help='Gap extension penalty (default: %(default)s)',
						type=float, default=0, dest='GAP_EXT')

	# misc

	parser.add_argument('--verbose', help='Be verbose (default: %(default)s)', action='store_true', default=False)
	
	parser.add_argument('-workers', help='Number of CPU workers (default: %(default)s)',
						type=int, default=10, dest='MAX_WORKERS')	

	parser.add_argument('-sigma_factor', help='The Sigma factor defines the greediness of the local alignment search procedure (default: %(default)s)',
						type=float, default=2, dest='SIGMA_FACTOR')	

	#parser.add_argument('-bfactor', help='bfactor (default: %(default)s)',
	#					 type=int, default=3, choices=range(1,4), metavar="[1-3]", dest='BF')
	
	#parser.add_argument('-emb_pool', help='embedding type (default: %(default)s) ',
	#					type=int, default=1, dest='EMB_POOL', choices=[1, 2, 4])

	args = parser.parse_args()
	
	# validate provided parameters
	assert args.MAX_WORKERS > 0
	
	assert args.MIN_SPAN_LEN >= args.WINDOW_SIZE, 'The minimum alignment length must be equal to or greater than the window length'
	
	if not args.mqsf and not args.mqmf: args.mqsf = True

	if args.mqsf:
		if os.path.isdir(args.output):
			raise ValueError("The provided output path points to a directory, a file was expected")
		elif not '.' in args.output:
			raise ValueError("The file name was not provided or it has no extension")
		elif os.path.exists(args.output) and os.path.isfile(args.output):
			raise ValueError("A file with this name already exists")

	elif args.mqmf:
		if not os.path.isdir(args.output):
			raise ValueError("The provided output directory does not exist")
		
	return args