import os
import argparse

import psutil


def get_available_cores() -> int:
	# chat gpt3 answer
	try:
		if "SLURM_JOB_CPUS_PER_NODE" in os.environ:
			num_cores = int(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
		else:
			num_cores = psutil.cpu_count(logical=False)
	except Exception as e:
		print(f'Cannot check available cores using safe value 4: {e}')
		num_cores = 4
	return num_cores



def range_limited_float_type(arg, MIN, MAX):
	""" Type function for argparse - a float within some predefined bounds """
	try:
		f = float(arg)
	except ValueError:
		raise argparse.ArgumentTypeError("Must be a floating point number")
	if f < MIN or f >= MAX :
		raise argparse.ArgumentTypeError(f"Argument must be {MIN} <= x < {MAX}")
	return f


def get_parser() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description =  
		"""pLM-BLAST: Searches a database of protein embeddings with a query protein embedding""",
		formatter_class=argparse.RawDescriptionHelpFormatter
		)

	range01 = lambda f:range_limited_float_type(f, 0, 1)
	range0100 = lambda f:range_limited_float_type(f, 0, 100)

	# input and output
	parser.add_argument('db', help=\
					 	'''Directory with a database to search, script will require db to be directory with embeddings and db[.fas, .csv, .p, .pkl] file with sequences.
						   For instance path: path/to/database will search for path/to/database directory and path/to/database[.fas, .csv, .p, .pkl] file''',
						type=str)	
	parser.add_argument('query', help=\
						'''Base name of query files. Extensions are automatically added, similar as db argument, additinally query will also look for .pt file
						   apart of directory with embeddings
						''',
						type=str)	
	parser.add_argument('output', help='Output csv file (or directory if `--separate` flag is given), results are stored with separator `;`',
						type=str)	
	parser.add_argument('--separate', help='Store the results of multi-query searches in separate files specified in `output`. Otherwise a single file is written',
					 action='store_true', default=False)
	parser.add_argument('--raw', help='Skip post-processing steps and return pickled Pandas data frames with all alignments', 
			 			action='store_true', default=False)
	
	# cosine similarity scan
	parser.add_argument('-cosine_percentile_cutoff', help=\
					 'Percentile cutoff for chunk cosine similarity pre-screening (default: %(default)s). The lower the value, the more sequences will be passed through the pre-screening procedure and then aligned with the more accurate but slower pLM-BLAST',
						type=range0100, default=70, dest='COS_PER_CUT')	
	parser.add_argument('--use_chunks', help=\
					 'Use fast chunk cosine similarity screening instead of regular cosine similarity screening. (default: %(default)s)',
			 action='store_true', default=False)
	
	# plmblast
	parser.add_argument('-alignment_cutoff', help='pLM-BLAST alignment score cut-off (default: %(default)s)',
						type=range01, default=0.3)						
	parser.add_argument('-win', help='Window length (default: %(default)s)',
						type=int, default=15, choices=range(50), metavar="[1-50]", dest='window_size')	
	parser.add_argument('-span', help='Minimal alignment length (default: %(default)s). Must be greater than or equal to the window length',
						type=int, default=25, choices=range(50), metavar="[1-50]", dest='min_spanlen')
	parser.add_argument('--global_aln', help='Use global pLM-BLAST alignment mode. (default: %(default)s)',
                    	default=False, action='store_true')
	parser.add_argument('-gap_ext', help='Gap penalty (default: %(default)s)',
						type=float, default=0, dest='gap_penalty')
	parser.add_argument('-bfactor', default=2, type=int, help= \
					 'increasing this value above 1 will reduce number of alignments that are very close to each other also increase search speed')
	parser.add_argument('--enh', default=False, action='store_true', help=\
					 """
					 use additional normalisation introduced in paper:  Embedding-based alignment: combining protein language
					 models and alignment approaches to detect structural
					 similarities in the twilight-zone link: https://www.biorxiv.org/content/10.1101/2022.12.13.520313v2.full.pdf
					 """)
	
	# misc
	parser.add_argument('--verbose', help='Be verbose (default: %(default)s)', action='store_true', default=False)
	parser.add_argument('-workers', help='Number of CPU workers (default: %(default)s) set 0 to use all available cores or num of cores set in slurm session',
						type=int, default=0, dest='workers')
	parser.add_argument('-sigma_factor', help='The Sigma factor defines the greediness of the local alignment search procedure (default: %(default)s)',
						type=float, default=2.5)	
	args = parser.parse_args()
	
	# validate provided parameters
	assert args.workers >= 0
	assert args.min_spanlen >= args.window_size, 'The minimum alignment length must be equal to or greater than the window length'
	# get available cores
	if args.workers == 0:
		args.workers = get_available_cores()
		print(f"using all {args.workers} CPU cores")
	return args