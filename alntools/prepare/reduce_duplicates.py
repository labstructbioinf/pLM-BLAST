

def reduce_duplicates_query_filedict(query_filedict):
	for key, value in query_filedict.items():
		for k in list(value.keys()):
			if k < key and query_filedict.get(key).get(k) != None:
				del query_filedict[key][k]
			if k == key:
				continue

	return query_filedict
