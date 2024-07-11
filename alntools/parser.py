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

	# Input and Output

	parser.add_argument('db', 
						help=('A database to be searched. The "db" argument defines a directory containing embedding files, '
							  'and a corresponding file with sequences (extensions .fas, .csv, .p, .pkl are automatically identified). '
							  'For example, if the database is located at /path/to/database, the script will search for the directory '
							  '/path/to/database and the file /path/to/database[.fas, .csv, .p, .pkl].'), 
						type=str)
				
	parser.add_argument('query', 
						help=('Query sequence(s). For a single query, the script expects files named query.fas and query.pt '
							  'containing the sequence and embedding, respectively. For multi-query searches, use the same format as '
							  'for the "db" argument (i.e., a directory with embeddings and a corresponding [.fas, .csv, .p, .pkl] '
							  'file with the sequence).'), 
						type=str)
				
	parser.add_argument('output', 
						help=('Output csv file (or directory if the `--separate` flag is given). Results are saved with separator `;`.'), 
						type=str)
				
	parser.add_argument('--separate', 
						help=('Store the results of multi-query searches in separate files specified in the "output" argument. '
							  'Otherwise, a single file will be written.'), 
						action='store_true', default=False)
				
	parser.add_argument('--raw', 
						help='Skip post-processing steps and return pickled Pandas data frames with all alignments.', 
						action='store_true', default=False)

	# Cosine Similarity Scan

	parser.add_argument('-cosine_percentile_cutoff', '-cpc', 
						help='Percentile cutoff for chunk cosine similarity pre-screening (default: %(default)s). '
							 'The lower the value, the more sequences will be pre-screened and then aligned with pLM-BLAST. '
							 'Setting the cutoff to 0 disables the pre-screening step.',
						type=range0100, default=70, dest='COS_PER_CUT')    

	parser.add_argument('--reduce_duplicates', 
						help='Filter redundant hits (feature under development, use with caution).',
						action='store_true', default=False)
	
	parser.add_argument('--only-scan', '-oc', 
					 help='run only prescreening, results will be stored in JSON format in path specified by `output` parameter\n'
					 	  "results format:\n"
						  'queryid1 : {'
						  '		{ file: targetfile1, score: scoreval1, condition: True }'
						  '     { file: targetfile2, score: scoreval2, condition: False }'
						  '}, queryid2 : {'
						  '	     { file: targetfile1, score: scoreval1, condition: True }'
						  '...'
						  '} Where score is a pre-screening value and condition checks whether quantile threshold criteria is met',
					 action='store_true',dest='only_scan', default=False)
	parser.add_argument('-cpc-kernel-size', dest='cpc_kernel_size', default=30)
	# pLM-BLAST

	parser.add_argument('-alignment_cutoff', 
						help='pLM-BLAST alignment score cut-off (default: %(default)s)',
						type=range01, default=0.3)    
	parser.add_argument('-win', 
						help='Window length (default: %(default)s)',
						type=int, default=15, choices=range(50), metavar="[1-50]", dest='window_size')    
	parser.add_argument('-span', 
						help='Minimal alignment length (default: %(default)s). Must be greater than or equal to the window length',
						type=int, default=25, choices=range(50), metavar="[1-50]", dest='min_spanlen')
	parser.add_argument('--global_aln', 
						help='Use global pLM-BLAST alignment mode. (default: %(default)s)',
						default=False, action='store_true')
	parser.add_argument('-gap_ext', 
						help='Gap penalty (default: %(default)s)',
						type=float, default=0.5, dest='gap_penalty')
	parser.add_argument('-bfactor', 
						default=2, type=int, 
						help='Increasing this value above 1 will reduce the number of alignments that are very close to each other, thus increasing the search speed.')
	parser.add_argument('--enh', 
						default=False, action='store_true', 
						help="""
							Use the additional normalization introduced in the paper https://doi.org/10.1093/bioinformatics/btad786 
							(feature under development, use with caution).
						""")
	# misc
	parser.add_argument('--verbose', help='Be verbose (default: %(default)s)', action='store_true', default=False)
	parser.add_argument('-workers', help='Number of CPU workers (default: %(default)s) Set to 0 to use all available cores or the number of cores set in a Slurm session.',
						type=int, default=0, dest='workers')
	parser.add_argument('-sigma_factor', help='The sigma factor defines the greediness of the local alignment search (default: %(default)s).',
						type=float, default=2.0)	
	args = parser.parse_args()
	
	# validate provided parameters
	assert args.workers >= 0
	if not args.only_scan:
		assert args.min_spanlen >= args.window_size, 'The minimum alignment length must be equal to or greater than the window length'
	else:
		print('running in pre-screening only mode')
	if args.reduce_duplicates and not args.enh:
		print('-reduce_duplicates flag is on but --enh is disabled, it will be turned on')
		args.enh = True
	# get available cores
	if args.workers == 0:
		args.workers = get_available_cores()
		print(f"using all {args.workers} CPU cores")
	return args