import json
from typing import List
from pathlib import Path


class PLMBlastAliasError(BaseException):
    pass

class PBAliasManager:
    '''
    index structure
    name: {
        raw: # human alias defintion
        indices: # indices of an alias
    }
    
    '''
    ext = ".alias.json"
    def __init__(self, dbpath: str | Path):
        if isinstance(dbpath, str):
            dbpath = Path(dbpath)
        self.dbpath = dbpath
        if not dbpath.is_dir():
            raise PLMBlastAliasError(f"no database at dir: {dbpath}")
        self.aliasfile = dbpath.with_suffix(self.ext)
        if not self.aliasfile.is_file():
            self.data = {}
        else:
            with self.aliasfile.open("rt") as fp:
                self.data = json.load(fp)
    
    def add(self, name: str, indices_string: str) -> None:
        # if already exsits
        if name in self.data:
            raise PLMBlastAliasError(f"alias with name: {name} already exists")
        else:
            indices = PBAliasManager.decode_indices(indices_string)
            print(f"registred new alias: {name} seqs: {len(indices)}")
            self.data[name] = {"indices": indices, "raw": indices_string}
            self._update()
            
    def remove(self, name: str):
        self._validate_alias(name)
        del self.data[name]
        self._update()
        
    def view(self):
        if self.data:
            print("no aliases registred yet")
        else:
            for name, idxdata in self.data.items():
                print(f"alias: {name} -> {idxdata['raw']}")
    
    def get(self, name: str) -> List[int]:
        self._validate_alias(name)
        return self.data[name]['indices']
        
    def _update(self):
        """save file with changes"""
        with self.aliasfile.open("wt") as fp:
            self.data = json.dump(self.data, fp, indent=4)
    
    @staticmethod
    def decode_indices(indices_string: str) -> List[int]:
        """decode sequence of indices in form of 1,2-201,204"""
        try:
            indices_groups = indices_string.split(",")
            indices = []
            for ig in indices_groups:
                if "-" not in ig: # single index
                    indices.append(int(ig))
                else:
                    start, stop = ig.split("-")
                    indices.extend(list(range(int(start), int(stop))))
            indices = list(set(indices))
            indices.sort()
        except Exception as e:
            raise BaseException(e)
        return indices
        
    def _validate_alias(self, name: str):
        if name == "":
            raise PLMBlastAliasError(
                "empty string passed as an alias name"
            )
        if name not in self.data:
            aliases_str = ", ".join(self.data.keys())
            raise PLMBlastAliasError(
                f"alias with name: {name} is not registred, available are: {aliases_str}")