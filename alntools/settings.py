mp_chunksize: int = 1

jobs_per_process: int = 30

SCR_BATCH_SIZE: int = 256
AVG_EMBEDDING_STD: float = 0.1

RESIDUES = list('ARNDCQEGHILKMFPSTWYVBZX*')
RESIDUE_GROUPS = ['GAVLI', 'FYW', 'CM', 'ST', 'KRH', 'DENQ', 'P', '-', 'X']
RESIDUE_GROUPMAP = {resgroup : i for i, resgroup in enumerate(RESIDUE_GROUPS)}

# changed from ".emb.64"
EMB64_EXT: str = "emb.64"
