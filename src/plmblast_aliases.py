import os
from pathlib import Path
import json
import argparse

from alntools.aliasmanager import PBAliasManager


class PLMBlastAliasError(BaseException):
    pass

def get_parser():
    
    parser = argparse.ArgumentParser(
        """
        manage db aliases
        example
        
        plmblast_aliases /path/to/db --view
        plmblast_aliases /path/to/db -add proteins1 1:200
        """)
    parser.add_argument('db', type=str)
    arggroups = parser.add_mutually_exclusive_group()
    arggroups.add_argument(
        "--view", 
        action="store_true",
        help="display aliases")
    arggroups.add_argument(
        "-add",
        nargs=2,
        help="add alias as a range of indices"
    )
    arggroups.add_argument(
        "-remove",
        help="remove single alias",
        type=str
    )
    return parser.parse_args()

def main():
    args = get_parser()
    dbdir = Path(args.db)
    aliasfile = dbdir.with_suffix(".json")
    print("looking for alias file", aliasfile)
    pbhandle = PBAliasManager(dbdir)
    if args.view:
        pbhandle.view()
    elif args.add:
        name, indices_string = args.add
        pbhandle.add(name, indices_string)
    elif args.remove:
        pbhandle.remove(args.name)
        
        
if __name__ == "__main__":
    main()