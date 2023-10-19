'''functions to make results more friendly'''
import argparse

import pandas as pd
from Bio.Align import substitution_matrices

from ..alignment import draw_alignment

blosum62 = substitution_matrices.load("BLOSUM62")

def calc_con(s1, s2):
	aa_list = list('ARNDCQEGHILKMFPSTWYVBZX*')
	res=[]
	for c1, c2 in zip(list(s1), list(s2)):
		if c1=='-' or c2=='-': 
			res+=' '
			continue
		bscore = blosum62[aa_list.index(c1)][aa_list.index(c2)]
		if bscore >= 6 or c1==c2:
			res+='|'
		elif bscore >= 0:
			res+='+'
		else:
			res+='.'
	return ''.join(res)
	

def calc_similarity(s1, s2) -> float:
	def aa_to_group(aa):
		for pos, g in enumerate(['GAVLI', 'FYW', 'CM', 'ST', 'KRH', 'DENQ', 'P', '-', 'X']):
			g = list(g)
			if aa in g: return pos
		# TODO be verbose
		assert False
	res = [aa_to_group(c1)==aa_to_group(c2) for c1, c2 in zip(list(s1), list(s2))]
	return sum(res)/len(res)
	

def calc_ident(s1, s2):
	res = [c1==c2 for c1, c2 in zip(list(s1), list(s2))]
	return sum(res)/len(res)
	

def prepare_output(args: argparse.Namespace,
				    resdf: pd.DataFrame,
					  query_id: str,
					    query_seq: str,
						db_df: pd.DataFrame) -> pd.DataFrame:
    '''
    add description to results
    '''
    # columns required
    COLUMNS_TO_USE = ['id', 'sequence']
    # columns to save
    COLUMNS_TO_SAVE = ['qid', 'score', 'ident', 'similarity', 'sid', 'sdesc', 'qstart',
                    'qend', 'qseq', 'con', 'tseq', 'tstart', 'tend', 'tlen', 'qlen',
                    'match_len']
	# checks
    for col in COLUMNS_TO_USE:
        if col not in db_df.columns:
            raise KeyError(f'missing {col} column in input dataframe')
    if len(resdf) == 0:
        print('No hits found!')
    else:
        resdf = resdf[resdf.score >= args.ALN_CUT]
        if len(resdf) == 0:
            print(f'No matches found! Try reducing the alignment_cutoff parameter. The current cutoff is {args.ALN_CUT}')
        else:
            # print('Preparing output...')
            resdf = resdf.drop(columns=['span_start', 'span_end', 'pathid', 'spanid', 'len'])            
            resdf['qid'] = query_id
            # TODO simplify above expressions
            resdf['sid'] = resdf['i'].apply(lambda i: db_df.iloc[i]['id'])
            if 'description' in db_df.columns:
                resdf['sdesc'] = resdf['i'].apply(lambda i: db_df.iloc[i]['description'].replace(';', ' '))
            else:
                resdf['sdecs'] = resdf['sid']
            resdf['tlen'] = resdf['i'].apply(lambda i: len(db_df.iloc[i]['sequence']))
            resdf['qlen'] = len(query_seq)
            resdf['qstart'] = resdf['indices'].apply(lambda i: i[0][1])
            resdf['qend'] = resdf['indices'].apply(lambda i: i[-1][1])
            resdf['tstart'] = resdf['indices'].apply(lambda i: i[0][0])
            resdf['tend'] = resdf['indices'].apply(lambda i: i[-1][0])
            resdf['match_len'] = resdf['qend'] - resdf['qstart'] + 1
            # TODO explain this
            assert all(resdf['qstart'].apply(lambda i: i <= len(query_seq) - 1))
            assert all(resdf['qend'].apply(lambda i: i <= len(query_seq) - 1))

            resdf.sort_values(by='score', ascending=False, inplace=True)
            resdf.reset_index(inplace=True)

            for idx, row in resdf.iterrows():
                tmp_aln = draw_alignment(row.indices,
                                        db_df.iloc[row.i].sequence,
                                        query_seq,
                                        output='str')
                tmp_aln = tmp_aln.split('\n')
				# add more descriptors
                resdf.at[idx, 'qseq'] = tmp_aln[2]
                resdf.at[idx, 'tseq'] = tmp_aln[0]
                resdf.at[idx, 'con'] = calc_con(tmp_aln[2], tmp_aln[0])
                resdf.at[idx, 'ident'] = round(calc_ident(tmp_aln[2], tmp_aln[0]), 2)
                resdf.at[idx, 'similarity'] = round(calc_similarity(tmp_aln[2], tmp_aln[0]), 2)

            resdf.drop(columns=['index', 'indices', 'i'], inplace=True)
            resdf.index.name = 'index'
            resdf = resdf[COLUMNS_TO_SAVE]
    return resdf