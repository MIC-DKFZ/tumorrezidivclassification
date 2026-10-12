import sqlite3
import time
import os


def migrate_file_paths(db_path: str, old_prefix: str, new_prefix: str):
    """
    Updates file paths in the database directly via SQLite engine for maximum efficiency.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"Starting migration: '{old_prefix}' -> '{new_prefix}'")
    start_time = time.time()

    try:
        # OPTIONAL BUT RECOMMENDED FOR MASSIVE UPDATES:
        # These pragmas drastically speed up large write operations.
        # WAL mode improves concurrency and write speed.
        cursor.execute("PRAGMA journal_mode = WAL;")
        # Turning off synchronous writes is safe *during* a standalone script run
        # (just don't power off your machine mid-run).
        cursor.execute("PRAGMA synchronous = OFF;")

        # The SQLite native update query.
        # 1. instr(file_path, ?) = 1 ensures we only match paths where the old_prefix is at the very beginning (index 1).
        # 2. SUBSTR(file_path, ?) slices the string starting right after the old prefix.
        # 3. || concatenates the new prefix with the sliced path.
        update_query = """
                       UPDATE file_properties
                       SET file_path = ? || SUBSTR(file_path, ?)
                       WHERE instr(file_path, ?) = 1; \
                       """

        # SQLite SUBSTR is 1-indexed. To get the rest of the string after old_prefix,
        # we start slicing at length(old_prefix) + 1
        slice_start_index = len(old_prefix) + 1

        # Execute the bulk update natively
        cursor.execute(update_query, (new_prefix, slice_start_index, old_prefix))

        # Get the number of rows affected
        rows_updated = cursor.rowcount

        # Commit the transaction
        conn.commit()

        elapsed_time = time.time() - start_time
        print(f"Success! Updated {rows_updated} rows in {elapsed_time:.3f} seconds.")

    except sqlite3.IntegrityError as e:
        # Since file_path is the PRIMARY KEY, this will trigger if replacing a prefix
        # results in a path that already exists in the database.
        print(f"Integrity Error (Collision): {e}")
        conn.rollback()
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        # Best practice to reset synchronous to NORMAL or FULL after the bulk operation
        cursor.execute("PRAGMA synchronous = NORMAL;")
        conn.close()


# code to merge 2 sqlite databases
def merge_sqlite_databases(db1_path: str | os.PathLike, db2_path: str | os.PathLike,
                           output_path: str | os.PathLike) -> None:
    """
    Merge two SQLite databases with the same schema into a new database.

    :param db1_path: Path to the first SQLite database
    :param db2_path: Path to the second SQLite database
    :param output_path: Path where the merged database will be saved
    """
    # Connect to the first database
    conn1 = sqlite3.connect(db1_path)
    cursor1 = conn1.cursor()

    # Connect to the second database
    conn2 = sqlite3.connect(db2_path)
    cursor2 = conn2.cursor()

    # Create a new database for the merged result
    if os.path.exists(output_path):
        os.remove(output_path)
    conn_out = sqlite3.connect(output_path)
    cursor_out = conn_out.cursor()

    # Get the list of tables (assuming both databases have the same schema)
    cursor1.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor1.fetchall()

    for table in tables:
        table_name = table[0]

        # Get the CREATE TABLE statement from the first database
        cursor1.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}';")
        create_table_stmt = cursor1.fetchone()[0]

        # Create the table in the output database
        cursor_out.execute(create_table_stmt)

        # Copy data from the first database
        cursor1.execute(f"SELECT * FROM {table_name}")
        rows = cursor1.fetchall()
        cursor_out.executemany(f"INSERT OR IGNORE INTO {table_name} VALUES ({','.join(['?' for _ in rows[0]])})", rows)

        # Copy data from the second database
        cursor2.execute(f"SELECT * FROM {table_name}")
        rows = cursor2.fetchall()
        cursor_out.executemany(f"INSERT OR IGNORE INTO {table_name} VALUES ({','.join(['?' for _ in rows[0]])})", rows)

    # Commit changes and close connections
    conn_out.commit()
    conn_out.execute("VACUUM")
    conn1.close()
    conn2.close()
    conn_out.close()


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":

    #to migrate database paths when mount points change
    DATABASE_FILE = "my_database.db"

    # Ensure trailing slashes are consistent!
    OLD_MOUNT = "/mnt/old_drive/dicom_data/"
    NEW_MOUNT = "/mnt/new_storage/dicom_data/"

    migrate_file_paths(DATABASE_FILE, OLD_MOUNT, NEW_MOUNT)

    # to merge 2 databases
    merge_sqlite_databases("db1.db", "db2.db", "merged_db.db")