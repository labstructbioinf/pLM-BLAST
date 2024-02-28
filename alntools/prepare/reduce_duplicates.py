
from typing import Dict

def reduce_duplicates_query_filedict(query_filedict: Dict[int, Dict[int, int]]) -> Dict[int, Dict[int, int]]:
    '''
    The function is used to simplify searching.
    The function skips the b vs a pairs, keeping only a vs b.
    Returns:
        dict: 
    '''
    assert isinstance(query_filedict, dict)
    assert len(query_filedict) > 0

    new_query_filedict = dict()

    for key, value in query_filedict.items():
        new_value = dict()
        for k, v in value.items():
            if k >= key and query_filedict.get(key).get(k) != None:
                new_value[k] = v

        if new_value:
            new_query_filedict[key] = new_value

    return new_query_filedict
