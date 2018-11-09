import numpy as np
import pandas as pd
import sqlite3
import argparse

if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        parser.add_argument('db', type=str, help='database')
        parser.add_argument('query', type=str, help='data base query [sql]')
        parser.add_argument('--model', type=int, default=3, help="model")

        args = parser.parse_args()

        conn = sqlite3.connect(args.db)

        with open(args.query) as q_file:
                data = pd.read_sql(q_file.read(), conn)
