import time
from datetime import datetime, timedelta
import pytz
import requests

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

def generate_output_brcp(uid, created_on):
    # Fetch, analyze, and upload data with enhanced error handling.
    try:
        # Fetch data from the database
        df = fetch_data_from_database(uid)
        if df is None or df.empty:
            error_msg = "Failed to fetch data from the database or DataFrame is empty."
            print(error_msg)
            return {"status": "Failed", "message": error_msg}

        final_df = analyse_data_using_gemini_for_brcp(df, uid, created_on)

        if final_df is None or final_df.empty:
            error_msg = "Data analysis failed. The output DataFrame is either missing or incorrect."
            print(error_msg)
            return {"status": "Failed", "message": error_msg}
        created_on_date = created_on.strftime('%Y-%m-%d')

        # Fetch interaction roster
        interaction_roster_df = fetchInteractionRoaster_forBrcp(created_on_date)
        # interaction_roster_df.to_csv('interationroaster.csv', index=False)

        # Merge dataframes on conversation ID
        interaction_roster_brcp_df = final_df.merge(
            interaction_roster_df,
            left_on="conversation_id",
            right_on="conversationid",
            how="left"
        )

        # Drop Unnamed columns (index columns) if present
        interaction_roster_brcp_df = interaction_roster_brcp_df.loc[
                                     :, ~interaction_roster_brcp_df.columns.str.contains('^Unnamed')
                                     ]

        # Drop 'conversationid' and 'agentemail1' columns if they exist
        drop_cols = [col for col in ['conversationid', 'agentemail1'] if col in interaction_roster_brcp_df.columns]
        interaction_roster_brcp_df = interaction_roster_brcp_df.drop(columns=drop_cols)

        # List of required columns
        required_columns = [
            "conversation_id", "request_id", "Sarcasm_rude_behaviour", "Sarcasm_rude_behaviour_evidence",
            "escalation_results", "Issue_Identification", "Probable_Reason_for_Escalation",
            "Probable_Reason_for_Escalation_Evidence", "Agent_Handling_Capability",
            "Wanted_to_connect_with_supervisor", "de_escalate", "Supervisor_call_connected",
            "call_back_arranged_from_supervisor", "supervisor_evidence", "Denied_for_Supervisor_call",
            "denied_evidence", "Today_Date", "uploaded_id", "Escalation_Category", "Location",
            "TL_Email_Id", "Email_Id", "Escalation_Keyword", "Short_Escalation_Reason"
        ]


        # Reorder columns: desired first, then rest
        current_cols = interaction_roster_brcp_df.columns.tolist()
        ordered_cols = [col for col in required_columns if col in current_cols]
        remaining_cols = [col for col in current_cols if col not in ordered_cols]
        final_column_order = ordered_cols + remaining_cols

        interaction_roster_brcp_df = interaction_roster_brcp_df[final_column_order]
        # Drop duplicate conversation IDs
        interaction_roster_brcp_df = interaction_roster_brcp_df.drop_duplicates(subset='conversation_id', keep='first')
        # Save to Excel with timestamped filename
        # interaction_roster_brcp_df.to_excel(f"interaction_roster_brcp_df_{created_on_date}.xlsx", index=False)
        # print("saved ")

        # Upload to database
        msg = upload_cred_result_on_database(interaction_roster_brcp_df, uid, created_on)
        if "successfully" in msg.lower():
            return {"status": "Success", "message": msg}
        else:
            error_msg = f"Uploading failed: {msg}"
            print(error_msg)
            return {"status": "Uploading Failed", "message": msg}

    except Exception as e:
        error_msg = f"Unexpected error in generate_output_brcp: {e}"
        print(error_msg)
        return {"status": "Error", "message": str(e)}


def get_brcp_result():
    """Fetch result from external API and process data using Gemini."""
    status, uid, created_on = is_latest_uid_present(INPUT_DATABASE)

    if status:
        print(f"No new uid found")
        return {"status": "Success", "message": "NO new ID found"}
    else:
        print(f"No transcript for {uid} UID. Generating transcripts for {uid}")
        print(f"{uid} UID is NOT present in tPrimaryInfo.")
    if uid:
        transmon_response = fetch_api_result(uid)
        gemini_response = generate_output_brcp(uid, created_on)
        status = {"TransmonResponse": transmon_response, "GeminiResponse": gemini_response}
        print(status)
    else:
        status = {"status": "Fetching latest Upload ID Failed", "message": "Upload Id not found"}
        print(status)

    return status