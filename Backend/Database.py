from dotenv import load_dotenv
import os
import pyodbc
import pandas as pd


class Database:
    def __init__(self):
        load_dotenv()
        self.server = os.getenv('SERVER')
        self.database = os.getenv('DATABASE')
        self.username = os.getenv('ADMINLOGIN')
        self.password = os.getenv('PASSWORD')
        self.driver = '{ODBC Driver 17 for SQL Server}'
        self.conn_str = f"DRIVER={self.driver};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password};Encrypt=yes;TrustServerCertificate=no;"

    def _connect(self):
        return pyodbc.connect(self.conn_str)

    def read(self, command):
        # Connect to the database
        conn = self._connect()

        # Execute query
        result = pd.read_sql(command, conn)

        conn.close()

        return result

    def write(self, command):
        # Connect to the database
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(command)
        cursor.commit()
        conn.close()


@staticmethod
def format_sql_command(table: str, data: dict) -> str:
    columns = ', '.join(data.keys())

    # Convert values to proper SQL representation, using NULL for None values and empty strings
    values = ', '.join('%s' if value is None or value == '' else f"'{value}'" if isinstance(
        value, str) else str(value) for value in data.values())

    # Create the parameterized query
    query = f'INSERT INTO {table} ({columns}) VALUES ({values});'
    return query
