import time

import pandas as pd

from fetchData import get_connection, INPUT_DATABASE, max_retries, retry_delay, OUTPUT_DATABASE


def fetcchBrcp(date):
    last_error = None

    for attempt in range(1, max_retries + 1):
        conn = get_connection(OUTPUT_DATABASE)
        if conn is None:
            last_error = "Database connection failed!"

            time.sleep(retry_delay * attempt)
            continue

        try:
            query = """
SELECT *
FROM brcpData 
WHERE CONVERT(DATE, TRY_PARSE(Today_Date AS DATETIME USING 'en-GB')) = ?
                """
            df = pd.read_sql_query(query, conn, params=(date,))
            df = df.drop_duplicates()

            if df.empty:
                print("No data found for the given UID.")
                return None

            print(f"Data Fetching Success: {df.shape[0]} conversation IDs, columns: {list(df.columns)}")
            return df

        except Exception as e:
            last_error = e
            time.sleep(retry_delay * attempt)

        finally:
            conn.close()

    # Only one message after all retries
    print(f"Data fetching failed after {max_retries} attempts.\nLast Error: {last_error}")
    return None

response = fetcchBrcp('2025-04-15')
print(response)