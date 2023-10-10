import os
import shutil
import pytest
import pandas as pd
import subprocess

DIR = os.path.dirname(__file__)

MAKE_INDEX_SCRIPT = os.path.join("scripts/makeindex.py")
EMB_SCRIPT = os.path.join("embeddings.py")
EMB64_SCRIPT = os.path.join("scripts/dbtofile.py")
PLMBLAST_SCRIPT = os.path.join("scripts/run_plm_blast.py")

PLMBLAST_DB = os.path.join(DIR, 'test_data/database')
PLMBLAST_DB_CSV = os.path.join(DIR, 'test_data/database.csv')

INPUT_FASTA_SINGLE = os.path.join(DIR, 'test_data/cupredoxin.fas')
INPUT_CSV_SINGLE = os.path.join(DIR, 'test_data/cupredoxin.csv')
INPUT_EMB_SINGLE = os.path.join(DIR, 'test_data/cupredoxin.pt')
OUTPUT_SINGLE = os.path.join(DIR, 'test_data/cupredoxin.hits.csv')

INPUT_FASTA_MULTI = os.path.join(DIR, 'test_data/multi_query.fas')
INPUT_CSV_MULTI = os.path.join(DIR, 'test_data/multi_query.csv')
INPUT_EMB_MULTI = os.path.join(DIR, 'test_data/multi_query.pt')
OUTPUT_MULTI = os.path.join(DIR, 'test_data/multi_query.hits.csv')

MULTI_QUERY_MULTI_FILE_PATH = os.path.join(DIR, 'test_data')

def clear_files(files):    
    for file in files:
        if os.path.isfile(file):
            os.remove(file)


@pytest.fixture(scope='session', autouse=True)
def remove_outputs():
	files = [
		INPUT_CSV_SINGLE, INPUT_EMB_SINGLE, OUTPUT_SINGLE,
		INPUT_CSV_MULTI, INPUT_EMB_MULTI, OUTPUT_MULTI,
		'tests/test_data/1BSV_1.hits.csv', 'tests/test_data/1FVK_1.hits.csv',
		'tests/test_data/1MXR_1.hits.csv', 'tests/test_data/7QZP_1.hits.csv']
	clear_files(files)
	shutil.rmtree(PLMBLAST_DB)
	

@pytest.mark.dependency()
def test_make_db_emb():
	proc = subprocess.run(["python", EMB_SCRIPT, PLMBLAST_DB_CSV, PLMBLAST_DB,
						'-embedder', 'pt', '-cname', 'sequence', '--gpu', '-bs', '0', '--asdir'],
		stderr=subprocess.PIPE,
		stdout=subprocess.PIPE)
	assert proc.returncode == 0, proc.stderr


@pytest.mark.dependency()
def test_make_db_emb64():
	proc = subprocess.run(["python", EMB64_SCRIPT, PLMBLAST_DB],
		stderr=subprocess.PIPE,
		stdout=subprocess.PIPE)
	assert proc.returncode == 0, proc.stderr


@pytest.mark.dependency()
def test_make_single_index():
	# Generate index csv for single query
	proc = subprocess.run(["python", MAKE_INDEX_SCRIPT, INPUT_FASTA_SINGLE, INPUT_CSV_SINGLE],
		stderr=subprocess.PIPE,
		stdout=subprocess.PIPE)
	assert proc.returncode == 0, proc.stderr


@pytest.mark.dependency(depends=['test_make_single_index'])
def test_make_single_emb():
	# Generate emb for single query
	proc = subprocess.run(["python", EMB_SCRIPT, 
						INPUT_CSV_SINGLE, INPUT_EMB_SINGLE,
						"-embedder", "pt", "-cname", "sequence", "--gpu"],
		stderr=subprocess.PIPE,
		stdout=subprocess.PIPE)
	assert proc.returncode == 0, proc.stderr


@pytest.mark.dependency()
@pytest.mark.parametrize('win', ["10", "20"])
@pytest.mark.parametrize('gap_ext', ["0", "0.1"])
def test_single_query(win: str, gap_ext: str):
	proc = subprocess.run(["python", PLMBLAST_SCRIPT, PLMBLAST_DB,
							INPUT_CSV_SINGLE[:-4], OUTPUT_SINGLE,
							"-win", win, '-gap_ext', gap_ext],
			stderr=subprocess.PIPE,
			stdout=subprocess.PIPE)
	# check process error code
	assert proc.returncode == 0, proc.stderr
	assert os.path.isfile(OUTPUT_SINGLE)
	output = pd.read_csv(OUTPUT_SINGLE, sep=";")
	assert output.shape == (28, 17)	or output.shape == (22, 17)

	if os.path.isfile(OUTPUT_SINGLE):
		os.remove(OUTPUT_SINGLE)


@pytest.mark.dependency()
def test_make_multi_index():
	# Generate index csv for multi query
	proc = subprocess.run(["python", MAKE_INDEX_SCRIPT, INPUT_FASTA_MULTI, INPUT_CSV_MULTI],
		stderr=subprocess.PIPE,
		stdout=subprocess.PIPE)
	assert proc.returncode == 0, proc.stderr


@pytest.mark.dependency(depends=['test_make_multi_index'])
def test_make_multi_embs():
	# Generate embs for multi query
	proc = subprocess.run(["python", EMB_SCRIPT, 
						INPUT_CSV_MULTI, INPUT_EMB_MULTI,
						"-embedder", "pt", "-cname", "sequence", "--gpu"],
		stderr=subprocess.PIPE,
		stdout=subprocess.PIPE)
	assert proc.returncode == 0, proc.stderr

@pytest.mark.dependency()
@pytest.mark.parametrize('win', ["10", "20"])
@pytest.mark.parametrize('gap_ext', ["0", "0.1"])
def test_multi_query(win: str, gap_ext: str):
	proc = subprocess.run(["python", PLMBLAST_SCRIPT, PLMBLAST_DB,
							INPUT_CSV_MULTI[:-4], OUTPUT_MULTI,
							"-win", win, '-gap_ext', gap_ext],
			stderr=subprocess.PIPE,
			stdout=subprocess.PIPE)
	# check process error code
	assert proc.returncode == 0, proc.stderr
	assert os.path.isfile(OUTPUT_MULTI)
	output = pd.read_csv(OUTPUT_MULTI, sep=";")
	assert output.shape == (410, 17) or output.shape == (318, 17)

	if os.path.isfile(OUTPUT_MULTI):
		os.remove(OUTPUT_MULTI)


@pytest.mark.dependency()
@pytest.mark.parametrize('win', ["10", "20"])
@pytest.mark.parametrize('gap_ext', ["0", "0.1"])
def test_multi_query_multi_files(win: str, gap_ext: str):
	proc = subprocess.run(["python", PLMBLAST_SCRIPT, PLMBLAST_DB,
							INPUT_CSV_MULTI[:-4], MULTI_QUERY_MULTI_FILE_PATH,
							"-win", win, '-gap_ext', gap_ext, '--mqmf'],
			stderr=subprocess.PIPE,
			stdout=subprocess.PIPE)
	# check process error code
	assert proc.returncode == 0, proc.stderr
