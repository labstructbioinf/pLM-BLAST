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
		"""pLM-BLAST: Searches a database of protein embeddings with a query protein embedding""",
		formatter_class=argparse.RawDescriptionHelpFormatter
		)

	range01 = lambda f:range_limited_float_type(f, 0, 1)
	range0100 = lambda f:range_limited_float_type(f, 0, 101)

	# input and output
	parser.add_argument('db', help='Directory with a database to search',
						type=str)	
	parser.add_argument('query', help='Base name of query files. Extensions are automatically added (.pt for embeddings and .csv, .p, .pkl, .fas or .fasta for sequences)',
						type=str)	
	parser.add_argument('output', help='Output file (or directory if `--separate` flag is given)',
						type=str)	
						
	parser.add_argument('--separate', help='Store the results of multi-query searches in separate files specified in `output`. Otherwise a single file is written',
					 action='store_true', default=False)
	parser.add_argument('--raw', help='Skip post-processing steps and return pickled Pandas data frames with all alignments', 
			 			action='store_true', default=False)
	
	# cosine similarity scan
	parser.add_argument('-cosine_percentile_cutoff', help=\
					 'Percentile cutoff for chunk cosine similarity pre-screening (default: %(default)s). The lower the value, the more sequences will be passed through the pre-screening procedure and then aligned with the more accurate but slower pLM-BLAST',
						type=range0100, default=95, dest='COS_PER_CUT')	
	parser.add_argument('--use_chunks', help=\
					 'Use fast chunk cosine similarity screening instead of regular cosine similarity screening. (default: %(default)s)',
			 action='store_true', default=False)
	
	# plmblast
	parser.add_argument('-alignment_cutoff', help='pLM-BLAST alignment score cut-off (default: %(default)s)',
						type=range01, default=0.3, dest='ALN_CUT')						
	parser.add_argument('-win', help='Window length (default: %(default)s)',
						type=int, default=10, choices=range(50), metavar="[1-50]", dest='WINDOW_SIZE')	
	parser.add_argument('-span', help='Minimal alignment length (default: %(default)s). Must be greater than or equal to the window length',
						type=int, default=25, choices=range(50), metavar="[1-50]", dest='MIN_SPAN_LEN')
	parser.add_argument('--global_aln', help='Use global pLM-BLAST alignment. (default: %(default)s)',
                    	default=False, action='store_true')
	parser.add_argument('-gap_ext', help='Gap extension penalty (default: %(default)s)',
						type=float, default=0, dest='GAP_EXT')
	# misc
	parser.add_argument('--verbose', help='Be verbose (default: %(default)s)', action='store_true', default=False)
	parser.add_argument('-workers', help='Number of CPU workers (default: %(default)s)',
						type=int, default=10, dest='MAX_WORKERS')
	parser.add_argument('-sigma_factor', help='The Sigma factor defines the greediness of the local alignment search procedure (default: %(default)s)',
						type=float, default=2, dest='SIGMA_FACTOR')	

	args = parser.parse_args()
	
	# validate provided parameters
	assert args.MAX_WORKERS > 0
	assert args.MIN_SPAN_LEN >= args.WINDOW_SIZE, 'The minimum alignment length must be equal to or greater than the window length'
	
	return args