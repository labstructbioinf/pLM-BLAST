import os
import json
import subprocess
import shutil

import pytest
import pandas as pd

DIR = os.path.dirname(__file__)
PLMBLAST_SCRIPT: str = 'scripts/run_plmblast.py'

def test_one_vs_all():

    subprocess.run('python scripts/run_plmblast.py' 'database' 'query' 'output.csv' '-use_chunks')