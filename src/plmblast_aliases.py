import os
from pathlib import Path
import json
import argparse


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
    parser.add_argument('db')
    arggroups = parser.add_mutually_exclusive_group()
    arggroups.add_argument(
        "--view", 
        action="store_true",
        description="display aliases")
    arggroups.add_argument(
        "-add",
        nargs=2,
        description="add alias as a range of indices"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_parser()
    dbdir = Path(args.db)
    aliasfile = dbdir.with_suffix(".json")
    print("looking for alias file", aliasfile)
    aliasfile.is_file():
        