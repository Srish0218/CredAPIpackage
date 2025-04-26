import os
from dotenv import load_dotenv
import pandas as pd
import pyodbc
from ZulipMessenger import reportSuccessMsgBRCP, reportError, reportStatus, reportSuccessMsgSoftSkill

load_dotenv()

# Read database credentials from environment variables
SERVER = os.getenv("DB_SERVER")
USERNAME = os.getenv("DB_USERNAME")
PASSWORD = os.getenv("DB_PASSWORD")
DRIVER = os.getenv("DB_DRIVER", "{ODBC Driver 17 for SQL Server}")
INPUT_DATABASE = os.getenv("INPUT_DATABASE")
OUTPUT_DATABASE = os.getenv("OUTPUT_DATABASE")

max_retries = 20
retry_delay = 5  # Seconds


def get_connection(DATABASE):
    """Establish a database connection with retries."""
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            return pyodbc.connect(
                f"DRIVER={DRIVER};SERVER={SERVER};DATABASE={DATABASE};UID={USERNAME};PWD={PASSWORD};"
            )
        except Exception as e:
            last_exception = e
            time.sleep(retry_delay * attempt)

    # Report error only once after all retries fail
    reportError(f"[Failed after {max_retries} attempts] Could not connect to {DATABASE}: {last_exception}")
    return None


import time

def upload_cred_result_on_database(final_df, uid, created_on, max_retries=3, retry_delay=5):
    """Insert DataFrame into the database with retry mechanism."""

    conn = get_connection(OUTPUT_DATABASE)
    if conn is None:
        return "Database connection failed!"

    if final_df.empty:
        msg = "No data to insert. DataFrame is empty."
        reportError(msg)
        return msg

    final_df = final_df.fillna(value="N/A")  # Handle NaN values once
    data_tuples = [tuple(row) for _, row in final_df.iterrows()]

    insert_query = """
    INSERT INTO brcpData (
        conversation_id, request_id, Sarcasm_rude_behaviour, Sarcasm_rude_behaviour_evidence,
        escalation_results, Issue_Identification, Probable_Reason_for_Escalation,
        Probable_Reason_for_Escalation_Evidence, Agent_Handling_Capability,
        Wanted_to_connect_with_supervisor, de_escalate, Supervisor_call_connected,
        call_back_arranged_from_supervisor, supervisor_evidence, Denied_for_Supervisor_call,
        denied_evidence, Today_Date, Uploaded_id, Escalation_Category, Location, TL_Email_Id ,Email_Id
    ) VALUES (?, ?, ?, ?,
        ?, ?, ?,
        ?, ?,
        ?, ?, ?,
        ?, ?, ?,
        ?, ?, ?, ? ,?, ?, ?)
    """

    last_error = None

    try:
        for attempt in range(1, max_retries + 1):
            try:
                cursor = conn.cursor()
                cursor.executemany(insert_query, data_tuples)
                conn.commit()
                reportSuccessMsgBRCP(uid, created_on)
                return "Data inserted successfully!"
            except Exception as e:
                conn.rollback()
                last_error = e
                reportError(f"Attempt {attempt}: Error inserting data - {e}")
                print(f"Attempt {attempt}: Error inserting data - {e}")

                if attempt < max_retries:
                    time.sleep(retry_delay)
        # If all attempts fail
        final_msg = f"Retry failed after {max_retries} attempts.\nLast Error: {last_error}"
        reportError(final_msg)
        return final_msg
    finally:
        conn.close()


def fetch_data_from_database(uid):
    """Fetch data from the database and return a DataFrame with retries."""
    last_error = None

    for attempt in range(1, max_retries + 1):
        conn = get_connection(INPUT_DATABASE)
        if conn is None:
            last_error = "Database connection failed!"
            time.sleep(retry_delay * attempt)
            continue

        try:
            query = """
                SELECT 
                    p.conversation_id, 
                    p.request_id, 
                    t.transcript
                FROM tPrimaryInfo p
                INNER JOIN tTranscript t
                    ON p.request_id = t.request_id
                WHERE p.uploaded_id = ?;
            """
            df = pd.read_sql_query(query, conn, params=(uid,))
            df = df.drop_duplicates()

            if df.empty:
                reportError("No data found for the given UID.")
                return None

            reportStatus(f"Data Fetching Success: {df.shape[0]} conversation IDs, columns: {list(df.columns)}")
            return df

        except Exception as e:
            last_error = e
            time.sleep(retry_delay * attempt)

        finally:
            conn.close()

    # Only one message after all retries
    reportError(f"Data fetching failed after {max_retries} attempts.\nLast Error: {last_error}")
    return None

def upload_softskill_result_on_database(df, date):
    """Insert DataFrame into the database with retries."""
    last_error = None  # Initialize the variable

    for attempt in range(1, max_retries + 1):
        conn = get_connection(OUTPUT_DATABASE)
        if conn is None:
            print(f"[Attempt {attempt}/{max_retries}] Failed to connect to the database.")
            time.sleep(retry_delay * attempt)
            continue

        try:
            cursor = conn.cursor()
            insert_query = """
                INSERT INTO softskill (
                    conversation_id, request_id, hold_request_found, hold_evidence,
                    CustomerLangCount, AgentLangCount, language_switch, Reassurance_result,
                    Reassurance_evidence, Apology_result, Apology_evidence, Empathy_result,
                    Empathy_evidence, No_Survey_Pitch, No_Survey_Pitch_Evidence, 
                    Unethical_Solicitation, Unethical_Solicitation_Evidence, DSAT_result, 
                    Customer_Issue_Identification, Reason_for_DSAT, Suggestion_for_DSAT_Prevention, 
                    DSAT_Category, Open_the_call_in_default_language, Open_the_call_in_default_language_evidence, 
                    Open_the_call_in_default_language_Reason, Hold_requested_before_dead_air, 
                    long_dead_air, dead_air_timestamp, VOC_Category, VOC_Core_Issue_Summary, 
                    timely_closing_result, timely_closing_evidence, hold_ended_in_required_duration, 
                    hold_ended_in_required_duration_evidence, hold_durations_after_hold_request, 
                    language_switch_result, Call_Opening_Category, default_opening_lang_Category, 
                    Apology_Category, Empathy_Category, Chat_Closing_Category, language_switch_category, 
                    Hold_category, Reassurance_Category, Language, Personalization_result, 
                    Personalization_Evidence, Delayed_call_opening, Delayed_call_opening_evidence, 
                    Further_Assistance, Further_Assistance_Evidence, Effective_IVR_Survey, 
                    Effective_IVR_Survey_Evidence, Branding, Branding_Evidence, Greeting, 
                    Greeting_Evidence, Greeting_the_customer, Greeting_the_customer_evidence, 
                    Self_introduction, Self_introduction_evidence, Identity_confirmation, 
                    Identity_confirmation_evidence, uploaded_date
                ) VALUES (
                ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?, 
                    ?, ?, ?, 
                    ?, ?, ?, 
                    ?, ?, ?, 
                    ?, ?, 
                    ?, ?, ?, ?, 
                    ?, ?, ?, 
                    ?, ?, 
                    ?, ?, ?, 
                    ?, ?, ?, ?, 
                    ?, ?, ?, ?, 
                    ?, ?, ?, 
                    ?, ?, ?, 
                    ?, ?, ?, ?, 
                    ?, ?, ?, 
                    ?, ?, ?, 
                    ?, ?
                )
            """
            df = df.fillna("N/A").astype(str)
            data_tuples = [tuple(row) for _, row in df.iterrows()]

            cursor.executemany(insert_query, data_tuples)
            conn.commit()
            reportSuccessMsgSoftSkill(date)
            return "Data inserted successfully!"

        except Exception as e:
            last_error = e
            conn.rollback()
            print(f"[Attempt {attempt}/{max_retries}] Data insertion failed! Error: {e}")
            time.sleep(retry_delay * attempt)

        finally:
            conn.close()

    final_msg = f"Data insertion failed after {max_retries} attempts.\nLast Error: {last_error}"
    reportError(final_msg)
    return final_msg


def fetch_data_softskill(date):
    """Fetch softskill-related data with retries."""
    for attempt in range(1, max_retries + 1):
        try:
            primary_conn = get_connection(INPUT_DATABASE)
            interaction_conn = get_connection(OUTPUT_DATABASE)

            primary_info_query = """
                SELECT * FROM tPrimaryInfo WHERE CONVERT(DATE, uploaded_on) = ?
            """
            primary_info_df = pd.read_sql(primary_info_query, primary_conn, params=[date])

            if primary_info_df.empty:
                return None, None, None, f"No data found for date {date} in tPrimaryInfo."

            request_ids_tuple = tuple(primary_info_df["request_id"].unique())
            conversation_id_tuple = tuple(primary_info_df["conversation_id"].unique())

            interaction_data_query = f"""
                SELECT conversationid, totalholdtime, calldisconnectionby, surveypoint 
                FROM interactiondb WHERE conversationid IN {str(conversation_id_tuple)}
            """
            transcript_query = f"SELECT * FROM tTranscript WHERE request_id IN {str(request_ids_tuple)}"
            transcriptchat_query = f"SELECT * FROM tutterances WHERE request_id IN {str(request_ids_tuple)}"

            interaction_data_df = pd.read_sql(interaction_data_query, interaction_conn)
            transcript_df = pd.read_sql(transcript_query, primary_conn)
            transcriptchat_df = pd.read_sql(transcriptchat_query, primary_conn)

            primary_info_df = primary_info_df.merge(
                interaction_data_df, left_on="conversation_id", right_on="conversationid", how="inner"
            ).drop(columns=["conversationid"])

            primary_info_df.drop_duplicates(subset=["request_id"], inplace=True)
            reportStatus(f"Data Fetching Success Softskill. Total Ids {primary_info_df.shape[0]}")
            return primary_info_df, transcript_df, transcriptchat_df, "Fetched Data Successfully"

        except Exception as e:
            reportError(f"[Attempt {attempt}/{max_retries}] Error: {e}")
            time.sleep(retry_delay * attempt)

        finally:
            for conn in [primary_conn, interaction_conn]:
                if conn:
                    conn.close()

    return None, None, None, "Data fetching failed after retries!"


def get_latest_uid(database):
    """Fetch the latest uploaded_id and created_on timestamp from Conversation_ID_List with retries."""
    for attempt in range(max_retries):
        conn = get_connection(database)
        if conn is None:
            return None, None  # Return None for both values if connection fails

        try:
            query = """
                SELECT uploaded_id, created_on FROM Conversation_ID_List 
                WHERE id = (SELECT MAX(id) FROM Conversation_ID_List);
            """
            cursor = conn.cursor()
            cursor.execute(query)
            row = cursor.fetchone()
            return (row[0], row[1]) if row else (None, None)  # Return both values
        except Exception as e:
            reportError(f"[ERROR] get_latest_uid failed (Attempt {attempt + 1}/3): {e}")
            time.sleep(retry_delay)
        finally:
            if conn:
                conn.close()

    return None, None  # Return None for both values if all attempts fail


def get_all_primaryinfo_uids():
    """Fetch all distinct uploaded_id values from tPrimaryInfo."""
    conn = get_connection(INPUT_DATABASE)
    if conn is None:
        return []
    try:
        query = "SELECT DISTINCT uploaded_id FROM tPrimaryInfo;"
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        return [row[0] for row in rows]
    except Exception as e:
        reportError(f"[ERROR] get_all_primaryinfo_uids: {e}")
        print(f"[ERROR] get_all_primaryinfo_uids: {e}")
        return []
    finally:
        conn.close()


def is_latest_uid_present(database):
    """Check if the latest UID from Conversation_ID_List is already in tPrimaryInfo."""
    latest_uid, created_on = get_latest_uid(database)
    if latest_uid is None:
        return False, latest_uid

    primaryinfo_uids = get_all_primaryinfo_uids()
    return latest_uid in primaryinfo_uids, latest_uid, created_on


def fetchInteractionRoaster_forBrcp(date, database=OUTPUT_DATABASE):
    conn = get_connection(database)
    if conn is None:
        return None

    interactionQuery = """
    SELECT conversationid, agentemail1
    FROM interactiondb 
    WHERE CONVERT(DATE, TRY_CAST(updated_at AS DATETIME)) = ?
    """
    interaction_df = pd.read_sql(interactionQuery, conn, params=[date])

    rosterQuery = """
    SELECT Location, TL_Email_Id, Email_ID 
    FROM ROSTER
    """
    roster_df = pd.read_sql(rosterQuery, conn)

    # Merge dataframes on agentmail1 from interactions and Email_ID from roster.
    interaction_roster_df = pd.merge(interaction_df, roster_df, left_on='agentemail1', right_on='Email_ID', how='inner')
    return interaction_roster_df



def get_created_on_by_uid(database, uid):
    """Fetch the created_on timestamp for a given uploaded_id from Conversation_ID_List."""
    for attempt in range(max_retries):
        conn = get_connection(database)
        if conn is None:
            return None  # Return None if connection fails

        try:
            query = """
                SELECT created_on FROM Conversation_ID_List 
                WHERE uploaded_id = ?;
            """
            cursor = conn.cursor()
            cursor.execute(query, (uid,))
            row = cursor.fetchone()
            return row[0] if row else None
        except Exception as e:
            reportError(f"[ERROR] get_created_on_by_uid failed (Attempt {attempt + 1}/3): {e}")
            time.sleep(retry_delay)
        finally:
            if conn:
                conn.close()

    return None  # Return None if all attempts fail


def is_uid_already_processed(uid):
    all_uids = get_all_primaryinfo_uids()
    return uid in all_uids



def fetchSoftskillOpsguru(date):
    for attempt in range(1, max_retries + 1):
        try:
            conn = get_connection(OUTPUT_DATABASE)

            if not conn:
                raise Exception("Database connection failed!")

            query = f"""
                SELECT * 
                FROM softskill WHERE CONVERT(DATE, TRY_CAST(uploaded_date AS DATETIME)) = ?
            """
            df = pd.read_sql_query(query, conn, params=(date,))

            return df, "Data Fetching Success"

        except Exception as e:
            reportError(f"[Attempt {attempt}/{max_retries}] Error: {e}")
            time.sleep(retry_delay * attempt)

        finally:
            if conn:
                conn.close()

    return None, "Data fetching failed after retries!"

def fetchBrcpOpsguru(date):
    for attempt in range(1, max_retries + 1):
        try:
            conn = get_connection(OUTPUT_DATABASE)

            if not conn:
                raise Exception("Database connection failed!")

            query = """
SELECT *
FROM brcpData 
WHERE CONVERT(DATE, TRY_PARSE(Today_Date AS DATETIME USING 'en-GB')) = ?
            """
            df = pd.read_sql_query(query, conn, params=(date,))

            return df, "Data Fetching Success"

        except Exception as e:
            reportError(f"[Attempt {attempt}/{max_retries}] Error: {e}")
            time.sleep(retry_delay * attempt)

        finally:
            if conn:
                conn.close()

    return None, "Data fetching failed after retries!"


def fetchInteractionOpsguru(date):
    for attempt in range(1, max_retries + 1):
        try:
            conn = get_connection(OUTPUT_DATABASE)

            if not conn:
                raise Exception("Database connection failed!")

            query = f"""
                SELECT * 
                FROM interactiondb WHERE CONVERT(DATE, TRY_CAST(updated_at AS DATETIME)) = ?
            """
            df = pd.read_sql_query(query, conn, params=(date,))

            return df, "Data Fetching Success"

        except Exception as e:
            reportError(f"[Attempt {attempt}/{max_retries}] Error: {e}")
            time.sleep(retry_delay * attempt)

        finally:
            if conn:
                conn.close()

    return None, "Data fetching failed after retries!"


def fetchRoster():
    for attempt in range(1, max_retries + 1):
        try:
            conn = get_connection(OUTPUT_DATABASE)

            if not conn:
                raise Exception("Database connection failed!")

            query = f"""
                SELECT * 
                FROM ROSTER 
            """
            df = pd.read_sql_query(query, conn)

            return df, "Data Fetching Success"

        except Exception as e:
            reportError(f"[Attempt {attempt}/{max_retries}] Error: {e}")
            time.sleep(retry_delay * attempt)

        finally:
            if conn:
                conn.close()

    return None, "Data fetching failed after retries!"
