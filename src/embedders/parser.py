import argparse

class EmbedderError(Exception):
	pass

def create_parser() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description =
		"""
		Embedding script create embeddings from sequences via desired embedder
		by default `seq` column in used as embedder input. Records are stored
		as list maintaining dataframe order.
		example use:
			embeddings start data.csv data.pt -cname seqfull
			# for fasta input
			embeddings start data.fasta data.pt
			# for file per embedding output
			embeddings start data.fasta /path/to/db --asdir 
            embeddings start data.fasta /path/to/db --npy
			# resume interrupted calculations
			embeddings resume /path/to/interrupted/db
		""",
		formatter_class=argparse.RawDescriptionHelpFormatter
		)
	parsers = parser.add_subparsers(title='options', required=True, dest='subparser_name')
	start_group = parsers.add_parser(name='start', help='starting new calculations')
	resume_group = parsers.add_parser(name='resume', help=\
	"""
	continue calculations from checkpoint, checkpoint is automatically created and stored as
	emb_checkpoint.json in output directory this is only available when using asdir flag
	""")
	resume_group.add_argument('output', type=str, help=\
						   'previous calculation directory or checkpoint file')
	start_group.add_argument('input', help=\
     'csv/pickle (.csv or .p) with `seq` column this can be changed by setting `-cname`',
						type=str)
	start_group.add_argument('output', help=\
		'''resulting file with list of embeddings or directory if `--asdir` is specified''',
						type=str)
	start_group.add_argument('-embedder', '-e', help=\
		"""
		name of the embedder by default `pt` - prot_t5_xl_half_uniref50-enc, `esm`
		for esm2_t33_650M_UR50D, `prost` for ProtT5-XL-U50 you can olso specify any model
		 supported by huggingface `AutoModel` typing `hf:modelname` (eg. `hf:Rostlab/prot_bert` 
		 for Rostlab/prot_bert, modelname may be also a path to pretrained model)
		""",
						dest='embedder', type=str, default='pt')
	start_group.add_argument('-cname', '-col', help='sequence column name for .csv inputs (default: %(default)s)',
						dest='cname', type=str, default='seq')
	start_group.add_argument('--cuda', '--gpu', help='if specified cuda device is used default False',
						dest='gpu', default=False, action='store_true')
	start_group.add_argument('-batch_size', '-b', '-bs', help=\
		'''batch size for loader longer sequences may require lower batch size set 0 to adaptive batch mode''',
						dest='batch_size', type=int, default=32)
	store_group = start_group.add_mutually_exclusive_group()
	store_group.add_argument('--asdir', help=\
		"""
		whether save output as directory where each embedding is a separate file,
		named as df index which is mandatory for large number of sequences
		""",
		action='store_true', default=False)
	store_group.add_argument('--h5py', 
		action='store_true', help=argparse.SUPPRESS, default=False)
	store_group.add_argument("--npy", action='store_true', default=False)
	start_group.add_argument('-truncate', '-t', default=1000, help=\
		"""
		cut sequences longer then parameter, similar to sequence[:truncate], helps to prevent OOM errors
		""",
		type=int, dest='truncate')
	start_group.add_argument('-res_per_batch', default=6000, type=int, help=\
		"""
		set the maximal number of residues in each batch, only used when batch_size is set to 0
		""")
	start_group.add_argument('--last_batch', help=argparse.SUPPRESS, type=int, default=0)
	start_group.add_argument('-nproc', '-np', help='number of process to spawn', default=1,
						  type=int)
	args = parser.parse_args()
	return args