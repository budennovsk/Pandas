import sqlite3
import csv
import pyodbc
from config import *

URL = 'E:/яндекс загрузка/file-csv.csv'
string_connect = f"""
Driver={DRIVER_NAME};
Server={SERVER_NAME};
Database={DATA_BASE};
Trusted_Connection=yes;
"""


def write_db_sql3(file_path):
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        with sqlite3.connect('sqldata.db') as con:
            cur = con.cursor()
            cur.execute("""Drop table if exists User""")
            cur.execute("""create table if not exists User
            (id INTEGER PRIMARY KEY AUTOINCREMENT,
            quarter TEXT,
            SER_REF TEXT,
            industry_code TEXT,
            industry_name TEXT,
            filled_jobs TEXT,
            filled_jobs_revised TEXT,
            filled_jobs_diff_q TEXT,
            filled_jobs_diff_n TEXT,
            total_earnings TEXT,
            total_earnings_revised TEXT,
            earnings_diff TEXT,
            earnings_diff_pr TEXT)
            """)
            data = []
            for idb in reader:
                data.append(
                    (idb[0], idb[1], idb[2], idb[3], idb[4], idb[5], idb[6], idb[7], idb[8], idb[9], idb[10], idb[11]))
            cur.executemany("""insert into User (
            quarter,
            SER_REF,
            industry_code,
            industry_name,
            filled_jobs,
            filled_jobs_revised,
            filled_jobs_diff_q,
            filled_jobs_diff_n,
            total_earnings,
            total_earnings_revised,
            earnings_diff,
            earnings_diff_pr)
            values (?, ?, ?,?, ?, ?,?, ?, ?,?,?,?)""", data)


def write_db_SQL(file_path, str_con):
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        with pyodbc.connect(str_con) as con:
            cur = con.cursor()
            cur.execute("""Drop table if exists User_table""")
            cur.execute("""CREATE TABLE User_table
            (id int IDENTITY(1, 1) PRIMARY KEY,
            quarter varchar(255),
            SER_REF varchar(255),
            industry_code varchar(255),
            industry_name varchar(255),
            filled_jobs varchar(255),
            filled_jobs_revised varchar(255),
            filled_jobs_diff_q varchar(255),
            filled_jobs_diff_n varchar(255),
            total_earnings varchar(255),
            total_earnings_revised varchar(255),
            earnings_diff varchar(255),
            earnings_diff_pr varchar(255))
            """)
            data = []
            for idb in reader:
                data.append(
                    (idb[0], idb[1], idb[2], idb[3], idb[4], idb[5], idb[6], idb[7], idb[8], idb[9], idb[10], idb[11]))
            cur.executemany("""insert into User_table (
            quarter,
            SER_REF,
            industry_code,
            industry_name,
            filled_jobs,
            filled_jobs_revised,
            filled_jobs_diff_q,
            filled_jobs_diff_n,
            total_earnings,
            total_earnings_revised,
            earnings_diff,
            earnings_diff_pr)
            values (?, ?, ?,?, ?, ?,?, ?, ?,?,?,?)""", data)


if __name__ == "__main__":
    write_db_sql3(URL)
    write_db_SQL(URL, string_connect)
