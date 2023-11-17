import os
import shutil
import pytest
import pandas as pd
import subprocess

from alntools.filehandle import BatchLoader

DIR = os.path.dirname(__file__)

EMB_SCRIPT = os.path.join("embeddings.py")
EMB64_SCRIPT = os.path.join("scripts/dbtofile.py")
PLMBLAST_SCRIPT = os.path.join("scripts/plmblast.py")

# use static db location for speed
PLMBLAST_DB = "/home/nfs/kkaminski/PLMBLST/ecod30db_20220902"
PLMBLAST_DB_CSV = PLMBLAST_DB + ".csv"

INPUT_FASTA_SINGLE = os.path.join(DIR, 'test_data/cupredoxin.fas')
INPUT_EMB_SINGLE = os.path.join(DIR, 'test_data/cupredoxin.pt')
OUTPUT_SINGLE = os.path.join(DIR, 'test_data/cupredoxin.hits.csv')

INPUT_FASTA_MULTI = os.path.join(DIR, 'test_data/rossmanns.fas')
INPUT_EMB_MULTI = os.path.join(DIR, 'test_data/rossmanns.pt')
OUTPUT_MULTI = os.path.join(DIR, 'test_data/multi_query.hits.csv')

MULTI_QUERY_MULTI_FILE_PATH = os.path.join(DIR, 'test_data')

def clear_files(files):    
    for file in files:
        if os.path.isfile(file):
            os.remove(file)


@pytest.fixture(scope='session', autouse=True)
def remove_outputs():
	files = [
		INPUT_EMB_SINGLE, OUTPUT_SINGLE, OUTPUT_MULTI,
		'tests/test_data/1BSV_1.hits.csv', 'tests/test_data/1FVK_1.hits.csv',
		'tests/test_data/1MXR_1.hits.csv', 'tests/test_data/7QZP_1.hits.csv']
	clear_files(files)
	if os.path.isdir(PLMBLAST_DB):
		shutil.rmtree(PLMBLAST_DB)


@pytest.mark.parametrize("infile", [INPUT_FASTA_SINGLE, INPUT_FASTA_MULTI])
@pytest.mark.parametrize("outfile", [INPUT_EMB_SINGLE, INPUT_EMB_MULTI])
def test_make_single_emb(infile, outfile):
	# Generate representations for further tests
	proc = subprocess.run(["python", EMB_SCRIPT, 'start',
						infile, outfile,
						"-embedder", "pt", "--gpu"],
		stderr=subprocess.PIPE,
		stdout=subprocess.PIPE)
	assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize('batch_size', [100, 200, 300])
def test_batch_loader_for_plmblast_loop(batch_size):
	query_seqs = {i : 'A'*123 for i in range(10)}
	filedict = dict()
	for qid in query_seqs:
		qid_files = {i : f'dump_{i}.txt'  for i in range(1000)}
		filedict[qid] = qid_files
	batchloader = BatchLoader(query_ids=list(query_seqs.keys()),
						   query_seqs=list(query_seqs.values()),
						   filedict=filedict,
						   batch_size=batch_size,
						   mode='file')
	
	query_files = {qid: list() for qid in query_seqs}
	assert len(batchloader) > 0
	assert len(batchloader) >= len(query_seqs)

	for qid, qseq, files in batchloader:
		assert len(files) != 0
		assert len(files) <= batch_size
		# files should not repeat within qid
		assert len(set(query_files[qid]) & set(files)) == 0
		query_files[qid].extend(files)

	# check if all files exists
	for qid, files in query_files.items():
		assert len(files) == len(filedict[qid])

@pytest.mark.dependency()
@pytest.mark.parametrize('win', ["10", "20", "30"])
@pytest.mark.parametrize('gap_ext', ["0", "0.1"])
def test_single_query(win: str, gap_ext: str):
	proc = subprocess.run(["python", PLMBLAST_SCRIPT, PLMBLAST_DB,
							INPUT_FASTA_SINGLE[:-4], OUTPUT_SINGLE,
							"-win", win, '-gap_ext', gap_ext],
			stderr=subprocess.PIPE,
			stdout=subprocess.PIPE)
	# check process error code
	assert proc.returncode == 0, proc.stderr
	assert os.path.isfile(OUTPUT_SINGLE)
	output = pd.read_csv(OUTPUT_SINGLE, sep=";")
	assert output.shape[0] > 0
	if os.path.isfile(OUTPUT_SINGLE):
		os.remove(OUTPUT_SINGLE)


@pytest.mark.dependency()
def test_make_multi_embs():
	# Generate embs for multi query
	proc = subprocess.run(["python", EMB_SCRIPT, 
						INPUT_FASTA_MULTI, INPUT_EMB_MULTI,
						"-embedder", "pt", "--gpu"],
		stderr=subprocess.PIPE,
		stdout=subprocess.PIPE)
	assert proc.returncode == 0, proc.stderr

@pytest.mark.dependency()
@pytest.mark.parametrize('win', ["10", "20"])
@pytest.mark.parametrize('gap_ext', ["0", "0.1"])
def test_multi_query(win: str, gap_ext: str):

	assert os.path.isfile(INPUT_FASTA_MULTI)
	assert os.path.isfile(INPUT_EMB_MULTI)

	proc = subprocess.run(["python", PLMBLAST_SCRIPT, PLMBLAST_DB,
							INPUT_FASTA_MULTI[:-4], OUTPUT_MULTI,
							"-win", win, '-gap_ext', gap_ext],
			stderr=subprocess.PIPE,
			stdout=subprocess.PIPE)
	# check process error code
	assert proc.returncode == 0, proc.stderr
	assert os.path.isfile(OUTPUT_MULTI)
	output = pd.read_csv(OUTPUT_MULTI, sep=";")
	assert output.shape[0] > 0

	if os.path.isfile(OUTPUT_MULTI):
		os.remove(OUTPUT_MULTI)


@pytest.mark.dependency()
@pytest.mark.parametrize('win', ["10", "20"])
@pytest.mark.parametrize('gap_ext', ["0", "0.1"])
def test_multi_query_multi_files(win: str, gap_ext: str):
	proc = subprocess.run(["python", PLMBLAST_SCRIPT, PLMBLAST_DB,
							INPUT_FASTA_MULTI[:-4], MULTI_QUERY_MULTI_FILE_PATH,
							"-win", win, '-gap_ext', gap_ext, '--separate'],
			stderr=subprocess.PIPE,
			stdout=subprocess.PIPE)
	# check process error code
	assert proc.returncode == 0, proc.stderr
