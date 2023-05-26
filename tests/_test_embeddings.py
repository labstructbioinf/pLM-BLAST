import os
import subprocess
import pytest

DIR = os.path.dirname(__file__)
EMBEDDING_SCRIPT = "embeddings.py"
EMBEDDING_DATA = os.path.join(DIR, "test_data/seq.p")
EMBEDDING_OUTPUT = os.path.join(DIR, "output/seq.emb")


@pytest.mark.parametrize("embedder", ["pt", "esm"])
@pytest.mark.parametrize("truncate", ["200", "500"])
def test_embedding_generation(embedder, truncate):
    if not os.path.isdir("tests/output"):
        os.mkdir("tests/output")
    proc = subprocess.run(["python", "embeddings.py",
    EMBEDDING_DATA, EMBEDDING_OUTPUT,
    "-embedder", embedder,
    "--truncate", truncate],
    stderr=subprocess.PIPE,
    stdout=subprocess.PIPE)
    # check output
    assert os.path.isfile(EMBEDDING_OUTPUT), f'missing embedding output file, {EMBEDDING_OUTPUT} {proc.stderr}'
    os.remove(EMBEDDING_OUTPUT)
    assert proc.returncode == 0, proc.stderr





