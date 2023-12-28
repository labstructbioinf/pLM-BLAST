'''functions to make results more friendly'''
from typing import List, Optional

import pandas as pd
from Bio.Align import substitution_matrices

from ..alignment import draw_alignment
from ..settings import (RESIDUES,
                        RESIDUE_GROUPMAP)

blosum62 = substitution_matrices.load("BLOSUM62")

COLUMNS_DB = ['id', 'sequence']
COLUMNS_QUERY = ['id', 'dbid', 'sequence']
# columns to save in output
COLUMNS_TO_SAVE = ['qid', 'score', 'ident', 'similarity', 'sid', 'qstart',
                'qend', 'qseq', 'con', 'tseq', 'tstart', 'tend', 'tlen', 'qlen',
                'match_len']

def calc_con(s1, s2):
	res = list()
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
					dbdf: pd.DataFrame,
                    alignment_cutoff: Optional[float] = None,
                    verbose: bool = False) -> pd.DataFrame:
    '''
    add description to results based on extracted alignments and database frame records
    
    Args:
        resdf (pd.DataFrame):
        dbdf (pd.DataFrame):
        alignment_cutoff: Optional (float) if > 0 results are filtred with this threshold
        verbose: (bool): 
    Returns:
        pd.DataFrame
    '''
    # drop technical columns    
    querydf = resdf.drop(columns=['span_start', 'span_end', 'spanid', 'len'])
    if 'index' in querydf.columns:
         querydf.drop(columns=['index'], inplace=True)
    if alignment_cutoff is None:
          alignment_cutoff = 0.0
	# check columns
    for col in COLUMNS_DB:
        if col not in dbdf.columns:
            raise KeyError(f'missing {col} column in input database frame')
    for col in COLUMNS_QUERY:
        if col not in querydf.columns:
              raise KeyError(f'missing {col} in input results frame')
    if len(querydf) == 0 and verbose:
        print('No hits found!')
        return pd.DataFrame()
    else:
        querydf = querydf[querydf.score >= alignment_cutoff].copy()
    if len(querydf) == 0:
        if verbose:
            print(f'No matches found for given query! Try reducing the alignment_cutoff parameter. The current cutoff is {alignment_cutoff}')
        return pd.DataFrame()
    else:
        querydf.rename(columns={'id' : 'qid'}, inplace=True)
        dbdf_matches = dbdf.iloc[querydf['dbid']].copy()
        aligmentlist: List[List[int, int]] = querydf['indices'].tolist()
        assert dbdf_matches.shape[0] == querydf.shape[0]

        querydf['sid'] = dbdf_matches['id'].values
        querydf['tlen'] = dbdf_matches['sequence'].apply(len).values.astype(int)
        querydf['qlen'] = querydf['sequence'].apply(len).values.astype(int)
        querydf['qstart'] =  [aln[0][1] for aln in aligmentlist]
        querydf['qend'] =  [aln[-1][1] for aln in aligmentlist]
        querydf['tstart'] = [aln[0][0] for aln in aligmentlist]
        querydf['tend'] = [aln[-1][0] for aln in aligmentlist]
        querydf['match_len'] = (querydf['qend'].values - querydf['qstart'].values + 1).astype(int)
        # check if alignment is not exeeding sequence lenght
        assert (querydf['qstart'] <= querydf['qlen']).all()
        assert (querydf['qstart'] <= querydf['qlen']).all()

        querydf.sort_values(by='score', ascending=False, inplace=True)

        alignment_desc = list()
        for _, row in querydf.iterrows():
            tmp_aln = draw_alignment(row.indices,
                                    dbdf.iloc[row.dbid].sequence,
                                    row.sequence,
                                    output='str')
            tmp_aln = tmp_aln.split('\n')
            qaln, taln = tmp_aln[2], tmp_aln[0]
            alignmnetdata = {
                    'qseq': qaln,
                    'tseq': taln,
                    'con': calc_con(qaln, taln),
                    'ident': calc_identity(qaln, taln),
                    'similarity': calc_similarity(qaln, taln)
                    }
            alignment_desc.append(alignmnetdata)
        alignment_desc = pd.DataFrame(alignment_desc, index=querydf.index)
        assert alignment_desc.shape[0] == querydf.shape[0]
        querydf = pd.concat((querydf, alignment_desc), axis=1)
        # drop parsed above columns
        if col in ['index', 'indices', 'i', 'dbid']:
                if col in querydf.columns:
                    querydf.drop(columns=col, inplace=True)
        querydf.index.name = 'index'
    # round alignment values
    querydf['score'] = querydf['score'].round(3)
    return querydf[COLUMNS_TO_SAVE]