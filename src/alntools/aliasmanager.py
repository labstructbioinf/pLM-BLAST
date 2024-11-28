import json
from pathlib import Path


class PLMBlastAliasError(BaseException):
    pass

class PBAliasManager:
    
    def __init__(self, dbpath: str | Path):
        if isinstance(dbpath, str):
            dbpath = Path(dbpath)
        self.dbpath = dbpath
        if not dbpath.is_dir():
            raise PLMBlastAliasError(f"no database at dir: {dbpath}")
        self.aliasfile = dbpath.with_suffix(".json")
        if not self.aliasfile.is_file():
            self.data = None
        else:
            with self.aliasfile.open("rt") as fp:
                self.data = json.load(fp)
    
    def add(self, name: str, indices_string: str):
        if name in self.data:
            raise PLMBlastAliasError(f"alias with name: {name} already exists")
    def view(self):
        
        
    
    @staticmethod
    def decode(indices_string):
        _splitted = indices_string.split(":")
        indices = None
        # example path:123-4154,143
        # split into 123-4154,143
        if len(_splitted) != 1:
            try:
                indices_groups = _splitted[1].split(",")
                indices = []
                for ig in indices_groups:
                    if "-" not in ig: # single index
                        indices.append(int(ig))
                    else:
                        start,stop = ig.split("-")
                        indices.extend(list(range(int(start), int(stop))))
                indices.sort()
            except Exception as e:
                raise BaseException("dump")
        