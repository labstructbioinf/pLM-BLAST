import os

extensions = ['.csv', '.p', '.pkl', '.fas', '.fasta']


def find_file_extention(infile: str) -> str:
    '''search for extension for query or index files'''
    assert isinstance(infile, str)
    infile_with_ext = infile
    for ext in extensions:
        if os.path.isfile(infile + ext):
            infile_with_ext = infile + ext
            break
    if infile_with_ext == "":
        raise FileNotFoundError(f'no matching index file {infile}')
    return infile_with_ext