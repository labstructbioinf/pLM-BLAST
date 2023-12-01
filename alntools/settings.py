mp_chunksize = 1

jobs_per_process = 30

SCR_BATCH_SIZE = 256
AVG_EMBEDDING_STD = 0.1

RESIDUES = list('ARNDCQEGHILKMFPSTWYVBZX*')
RESIDUE_GROUPS = ['GAVLI', 'FYW', 'CM', 'ST', 'KRH', 'DENQ', 'P', '-', 'X']
RESIDUE_GROUPMAP = {resgroup : i for i, resgroup in enumerate(RESIDUE_GROUPS)}

EMB64_EXT = ".emb.64"