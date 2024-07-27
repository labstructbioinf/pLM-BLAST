'''test plmblast.py script in different ways'''
import os
import subprocess

import pytest
import pandas as pd

from alntools.filehandle import BatchLoader


DIR = os.path.dirname(__file__)
DIRUP = os.path.dirname(DIR)
SCRIPT = os.path.join("scripts/plmblast.py")
TESTDATA = os.environ.get("PLMBLAST_TESTDATA")
NUM_WORKERS = int(os.environ.get("PLMBLAST_WORKERS", 4))
# use static db location to avoid unessesery calculations
PLMBLAST_DB = os.environ.get("PLMBLAST_DB")
PLMBLAST_DB_CSV = PLMBLAST_DB + ".csv"
INPUT_SINGLE = os.path.join(TESTDATA, 'cupredoxin')
INPUT_MULTI = os.path.join(TESTDATA, 'rossmanns')
# outputs
OUTPUT_SINGLE = os.path.join(DIR, 'test_data/cupredoxin.hits.csv')
OUTPUT_MULTI = os.path.join(DIR, 'test_data/rossmanns.hits.csv')
MULTI_QUERY_MULTI_FILE_PATH = os.path.join(DIR, 'test_data')


@pytest.fixture(scope='session', autouse=True)
def remove_outputs():
	for file in [OUTPUT_MULTI, OUTPUT_SINGLE]:
		if os.path.isfile(file):
			os.remove(file)


def test_data_exists():
	assert os.path.isfile(SCRIPT)
	assert os.path.isdir(PLMBLAST_DB), f"missing db directory: {PLMBLAST_DB}"
	assert os.path.isfile(PLMBLAST_DB_CSV)
	for ext in [".fas", ".pt"]:
		assert os.path.isfile(INPUT_SINGLE + ext)
		assert os.path.isfile(INPUT_MULTI + ext)

# TODO change Batchloader input to DBData objects
#@pytest.mark.parametrize('batch_size', [100, 200, 300])
#def test_batch_loader_for_plmblast_loop(batch_size):
#	query_seqs = {i : 'A'*123 for i in range(10)}
#	filedict = dict()
#	for qid in query_seqs:
#		qid_files = {i : f'dump_{i}.txt'  for i in range(1000)}
#		filedict[qid] = qid_files
#	batchloader = BatchLoader(query_ids=list(query_seqs.keys()),
#						   query_seqs=list(query_seqs.values()),
#						   filedict=filedict,
#						   batch_size=batch_size,
#						   mode='file')
#	
#	query_files = {qid: list() for qid in query_seqs}
#	assert len(batchloader) > 0
#	assert len(batchloader) >= len(query_seqs)
#
#	for qid, qseq, files in batchloader:
#		assert len(files) != 0
#		assert len(files) <= batch_size
#		# files should not repeat within qid
#		assert len(set(query_files[qid]) & set(files)) == 0
#		query_files[qid].extend(files)
#
#	# check if all files exists
#	for qid, files in query_files.items():
#		assert len(files) == len(filedict[qid])


@pytest.mark.parametrize('win', [25])
@pytest.mark.parametrize('gap_ext', [0, 0.1])
@pytest.mark.parametrize("cosine_percentile_cutoff", [90, 0])
def test_single_query(win: int, gap_ext: int, cosine_percentile_cutoff: int):
	cmd = f"python {SCRIPT} {PLMBLAST_DB} {INPUT_SINGLE} {OUTPUT_SINGLE} -win {win} -gap_ext {gap_ext}"
	cmd += f" -cosine_percentile_cutoff {cosine_percentile_cutoff}"
	cmd += " -alignment_cutoff 0.2"
	proc = subprocess.run(cmd.split(" "), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
	# check process error code
	assert proc.returncode == 0, proc.stderr
	# check if there are hits
	assert os.path.isfile(OUTPUT_SINGLE), f"missing output after run from cmd: {proc.stdout}"
	output = pd.read_csv(OUTPUT_SINGLE, sep=";")
	if win < 20:
		assert output.shape[0] > 0, "no results for given query"


def test_results_reproducibility():
	result_stack = list()
	for n in range(3):
		output =  OUTPUT_SINGLE.replace(".csv", f"{n}.csv")
		cmd = f"python {SCRIPT} {PLMBLAST_DB} {INPUT_SINGLE} {output}"
		proc = subprocess.run(cmd.split(" "), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
		# check process error code
		assert proc.returncode == 0, proc.stderr
		result_stack.append(pd.read_csv(output))
	for i, resulti in enumerate(result_stack):
		for j, resultj in enumerate(result_stack):
			if i == j:
				continue
			assert (resulti == resultj).all(), 'results are not identical'


@pytest.mark.parametrize('win', [10, 15])
@pytest.mark.parametrize('gap_ext', [0, 0.1])
def test_multi_query(win: str, gap_ext: str):
	cmd = f"python {SCRIPT} {PLMBLAST_DB} {INPUT_MULTI} {OUTPUT_MULTI} -win {win} -gap_ext {gap_ext}"
	proc = subprocess.run(cmd.split(" "), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
	# check process error code
	assert proc.returncode == 0, proc.stderr
	if not os.path.isfile(OUTPUT_MULTI):
		raise FileNotFoundError(f'missing output after plmblast run, err: {proc.stderr}')
	output = pd.read_csv(OUTPUT_MULTI, sep=";")
	assert output.shape[0] > 0


@pytest.mark.parametrize('win', [10, 20])
@pytest.mark.parametrize('gap_ext', [0, 0.1])
def test_multi_query_multi_files(win: str, gap_ext: str):
	cmd = f"python {SCRIPT} {PLMBLAST_DB} {INPUT_MULTI} {OUTPUT_MULTI} -win {win} -gap_ext {gap_ext} --separate"
	proc = subprocess.run(cmd.split(" "), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
	# check process error code
	if proc.returncode != 0:
		raise OSError(proc.stderr)


def test_self_similarity():
	cmd = f"python {SCRIPT} {INPUT_SINGLE} {INPUT_SINGLE} {OUTPUT_SINGLE}"
	proc = subprocess.run(cmd.split(" "), stderr=subprocess.PIPE, stdout=subprocess.PIPE)
	# check process error code
	if proc.returncode != 0:
		raise OSError(proc.stderr)
	assert os.path.isfile(OUTPUT_SINGLE)
	output = pd.read_csv(OUTPUT_SINGLE)
	# self similarity will always be not empty
	assert output.shape[0] > 0