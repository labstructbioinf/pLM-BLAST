mp_chunksize: int = 1

jobs_per_process: int = 30

SCR_BATCH_SIZE: int = 256
AVG_EMBEDDING_STD: float = 0.1

RESIDUES = list('ARNDCQEGHILKMFPSTWYVBZX*')
RESIDUE_GROUPS = ['GAVLI', 'FYW', 'CM', 'ST', 'KRH', 'DENQ', 'P', '-', 'X']
RESIDUE_GROUPMAP = {resgroup : i for i, resgroup in enumerate(RESIDUE_GROUPS)}

# changed from ".emb.64"
EMB64_EXT: str = "emb.64"

# available index file extensions
EXTENSIONS = ['.csv', '.p', '.pkl', '.fas', '.fasta']

# ANSI escape sequences for colors
colors = {
	'black': '\033[30m',
	'red': '\033[31m',
	'green': '\033[32m',
	'yellow': '\033[33m',
	'blue': '\033[34m',
	'magenta': '\033[35m',
	'cyan': '\033[36m',
	'white': '\033[37m',
	'reset': '\033[0m'  # Reset to default color
	}


# columns to save in output file
COLUMNS_TO_SAVE = ['qid', 'score', 'ident', 'similarity', 'sid', 'qstart',
                'qend', 'qseq', 'con', 'tseq', 'tstart', 'tend', 'tlen', 'qlen',
                'match_len']

COLUMNS_TO_SAVE_OPTIONAL = ['sdesc']