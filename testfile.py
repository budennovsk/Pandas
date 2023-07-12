import os
import pyodbc
from config import *
# pth = 'E:/Тестовые задания/'
# arr = os.listdir(pth)
# print(arr,'ddd')
# for i in arr:
#     print(i)
#
# car = os.walk(pth)
# for r, d,f in car:
#     print(f)

conn = pyodbc.connect(f"""
Driver={DRIVER_NAME};
Server={SERVER_NAME};
Database={DATA_BASE};
Trusted_Connection=yes;
""")

cur = conn.cursor()
print(cur)
cur.execute('SELECT id FROM Table_1')
v = cur.fetchone()
print(v.id)
b = cur.execute("""CREATE TABLE asfdf
(name VARCHAR(50),  age INTEGER)""")
b.commit()
conn.close()
