'''functions to make results more friendly'''
import argparse
from typing import List, Optional

import pandas as pd
from Bio.Align import substitution_matrices

from ..alignment import draw_alignment

blosum62 = substitution_matrices.load("BLOSUM62")

RESIDUES = list('ARNDCQEGHILKMFPSTWYVBZX*')
COLUMNS_DB = ['id', 'sequence']
COLUMNS_QUERY = ['id', 'dbid', 'sequence']
# columns to save in output
COLUMNS_TO_SAVE = ['qid', 'score', 'ident', 'similarity', 'sid', 'sdesc', 'qstart',
                'qend', 'qseq', 'con', 'tseq', 'tstart', 'tend', 'tlen', 'qlen',
                'match_len']
RESIDUE_GROUPS = ['GAVLI', 'FYW', 'CM', 'ST', 'KRH', 'DENQ', 'P', '-', 'X']
RESIDUE_GROUPMAP = {resgroup : i for i, resgroup in enumerate(RESIDUE_GROUPS)}



def calc_con(s1, s2):
	res=[]
	for c1, c2 in zip(list(s1), list(s2)):
		if c1=='-' or c2=='-': 
			res+=' '
			continue
		bscore = blosum62[RESIDUES.index(c1)][RESIDUES.index(c2)]
		if bscore >= 6 or c1==c2:
			res+='|'
		elif bscore >= 0:
			res+='+'
		else:
			res+='.'
	return ''.join(res)

def residue_to_group(residue: str) -> int:	
    for resgroup, groupid in RESIDUE_GROUPMAP.items():
        if residue in resgroup:
            return groupid
    assert False, f'invalid resdue {residue}'

def calc_similarity(s1, s2) -> float:
	res = [residue_to_group(c1)==residue_to_group(c2) for c1, c2 in zip(list(s1), list(s2))]
	return round(sum(res)/len(res), 2)
	

def calc_identity(s1: List[str], s2: List[str]) -> float:
    '''
    return identity based on alignment string
    '''
    res = [c1==c2 for c1, c2 in zip(list(s1), list(s2))]
    return round(sum(res)/len(res), 2)
	

def prepare_output(resdf: pd.DataFrame,
					db_df: pd.DataFrame,
                    alignment_cutoff: Optional[float] = None,
                    verbose: bool = False) -> pd.DataFrame:
    '''
    add description to results based on extracted alignments and database frame records
    
    Args:
        resdf (pd.DataFrame):
        db_df (pd.DataFrame):
        alignment_cutoff: Optional (float) if > 0 results are filtred with this threshold
        verbose: (bool):

    '''
    if alignment_cutoff is None:
          alignment_cutoff = 0.0
	# check columns
    for col in COLUMNS_DB:
        if col not in db_df.columns:
            raise KeyError(f'missing {col} column in input database frame')
    for col in COLUMNS_QUERY:
        if col not in resdf.columns:
              raise KeyError(f'missing {col} in input results frame')
    if len(resdf) == 0:
        print('No hits found!')
    else:
        resdf = resdf[resdf.score >= alignment_cutoff]
        if len(resdf) == 0:
            print(f'No matches found! Try reducing the alignment_cutoff parameter. The current cutoff is {alignment_cutoff}')
        else:
            # drop technical columns
            resdf.drop(columns=['span_start', 'span_end', 'pathid', 'spanid', 'len'], inplace=True)            
            resdf.rename(columns={'id' : 'qid'}, inplace=True)
            # TODO simplify below expressions 
            # add database description
            if 'description' in db_df.columns:
                resdf['sdesc'] = resdf['dbid'].apply(lambda i: db_df.iloc[i]['description'].replace(';', ' '))
            else:
                resdf['sdesc'] = resdf['dbid']
            resdf['sid'] = resdf['dbid'].apply(lambda i: db_df.iloc[i]['id'])
            resdf['tlen'] = resdf['dbid'].apply(lambda i: len(db_df.iloc[i]['sequence']))
            resdf['qlen'] = resdf['sequence'].apply(len)
            resdf['qstart'] = resdf['indices'].apply(lambda i: i[0][1])
            resdf['qend'] = resdf['indices'].apply(lambda i: i[-1][1])
            resdf['tstart'] = resdf['indices'].apply(lambda i: i[0][0])
            resdf['tend'] = resdf['indices'].apply(lambda i: i[-1][0])
            resdf['match_len'] = resdf['qend'] - resdf['qstart'] + 1
            # check if alignment is not exeeding sequence lenght
            assert (resdf['qstart'] <= resdf['qlen']).all()
            assert (resdf['qstart'] <= resdf['qlen']).all()

            resdf.sort_values(by='score', ascending=False, inplace=True)
            resdf.reset_index(inplace=True)

            alignment_descriptors = list()
            for idx, row in resdf.iterrows():
                tmp_aln = draw_alignment(row.indices,
                                        db_df.iloc[row.dbid].sequence,
                                        row.sequence,
                                        output='str')
                tmp_aln = tmp_aln.split('\n')
                alignmnetdata = {
                      'qseq': tmp_aln[2],
                      'tseq': tmp_aln[0],
                      'con': calc_con(tmp_aln[2], tmp_aln[0]),
                      'ident': calc_identity(tmp_aln[2], tmp_aln[0]),
                      'similarity': calc_similarity(tmp_aln[2], tmp_aln[0])
                      }
                alignment_descriptors.append(alignmnetdata)
            alignment_descriptors = pd.DataFrame(alignment_descriptors)
            resdf = pd.concat((resdf, alignment_descriptors), axis=1)
            
            # drop parsed above columns
            if col in ['index', 'indices', 'i', 'dbid']:
                  if col in resdf.columns:
                        resdf.drop(columns=col, inplace=True)
            resdf.index.name = 'index'
            resdf = resdf[COLUMNS_TO_SAVE]
    return resdf