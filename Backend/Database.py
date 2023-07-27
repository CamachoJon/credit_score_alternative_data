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

        # Split the command into separate commands at the first semicolon
        commands = command.split(";", 1)
        cursor.execute(commands[0].strip())
        cursor.commit()

        # Check if there is a second command to be executed (for fetching the ID)
        if len(commands) > 1 and commands[1].strip():
            # Execute the SELECT statement to get the ID of the last inserted row
            cursor.execute(commands[1].strip())

            # Fetch the ID
            result_tuple = cursor.fetchone()

            if result_tuple:
                # The ID is the first (and only) element of the tuple
                new_id = int(result_tuple[0])
                conn.close()
                return new_id
            else:
                conn.close()
                return None
        else:
            conn.close()

    def format_sql_command(table: str, data: dict) -> str:
        columns = ', '.join(data.keys())

        # Convert values to proper SQL representation, using NULL for None values and empty strings
        values = ', '.join('NULL' if value is None or value == '' or value == 'NaN' else f"'{value}'" if isinstance(
            value, str) else str(value) for value in data.values())

        if table == 'USERS':
            # Create the parameterized query
            query = f'INSERT INTO {table} ({columns}) VALUES ({values}); SELECT SCOPE_IDENTITY() AS LastID;'
        else:
            # Create the parameterized query
            query = f'INSERT INTO {table} ({columns}) VALUES ({values});'
            
        return query




