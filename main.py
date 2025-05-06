import time
from datetime import datetime, timedelta
import pytz
import requests
from fastapi import FastAPI
from ZulipMessenger import reportTranscriptGenerated, reportError, reportStatus
from analyseData import analyse_data_using_gemini_for_brcp, analyse_data_for_soft_skill
from fetchData import fetch_data_from_database, upload_cred_result_on_database, fetch_data_softskill, \
    is_latest_uid_present, INPUT_DATABASE, fetchInteractionRoaster_forBrcp, get_created_on_by_uid, \
    fetchSoftskillOpsguru, fetchBrcpOpsguru, fetchInteractionOpsguru, fetchRoster, uploadOpsgurudata
from resources.working_with_files import createDfOpsguru

app = FastAPI()


def fetch_api_result(uid: str, max_retries=100, retry_delay=5):
    """Fetch API result from external service with retry mechanism."""
    url = f"https://tmc.wyzmindz.com/CredASR/api/CredASR/FetchResultFromVocab?uploaded_id={uid}"
    headers = {"accept": "*/*"}
    errorCombined = []

    for attempt in range(1, max_retries + 1):
        start_time = time.time()

        try:
            response = requests.post(url, headers=headers)  # Added timeout for reliability
            end_time = time.time()
            response_time = round(end_time - start_time, 2)

            if response.status_code in [200, 204]:
                reportTranscriptGenerated(uid)
                return {"status": "Success", "message": f"Request successful in {response_time} seconds"}

            elif 500 <= response.status_code < 600:
                reportError(f"Attempt {attempt}: Server error {response.status_code}, retrying...")
            else:
                reportError(f"HTTP {response.status_code}: {response.text}")
                return {"status": "Failed", "error_code": response.status_code, "message": response.text}

        except requests.Timeout:
            print(f"Attempt {attempt}: API request timed out. Retrying in {retry_delay} seconds...")
            errorCombined.append(f"Attempt {attempt}: API request timed out. Retrying in {retry_delay} seconds...")
        except requests.ConnectionError:
            errorCombined.append(f"Attempt {attempt}: Connection error. Retrying in {retry_delay} seconds...")
            print(f"Attempt {attempt}: Connection error. Retrying in {retry_delay} seconds...")
        except requests.RequestException as e:
            errorCombined.append(str(e))
            print(f"Attempt {attempt}: Unexpected error - {e}")
            return {"status": "Error", "message": str(e)}

        if attempt < max_retries:
            time.sleep(retry_delay)
    error = {"status": "Failed", "message": f"API request failed after {max_retries} attempts",
             "ErrorStatement": errorCombined}
    reportError(error)

    return error


def generate_output_brcp(uid, created_on):
    # Fetch, analyze, and upload data with enhanced error handling.
    try:
        # Fetch data from the database
        df = fetch_data_from_database(uid)
        if df is None or df.empty:
            error_msg = "Failed to fetch data from the database or DataFrame is empty."
            reportError(error_msg)
            return {"status": "Failed", "message": error_msg}

        final_df = analyse_data_using_gemini_for_brcp(df, uid, created_on)

        if final_df is None or final_df.empty:
            error_msg = "Data analysis failed. The output DataFrame is either missing or incorrect."
            reportError(error_msg)
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
            reportError(error_msg)
            return {"status": "Uploading Failed", "message": msg}

    except Exception as e:
        error_msg = f"Unexpected error in generate_output_brcp: {e}"
        reportError(error_msg)
        return {"status": "Error", "message": str(e)}


@app.get("/brcp")
def get_brcp_result():
    """Fetch result from external API and process data using Gemini."""
    status, uid, created_on = is_latest_uid_present(INPUT_DATABASE)

    if status:
        print(f"No new uid found")
        return {"status": "Success", "message": "NO new ID found"}
    else:
        reportStatus(f"No transcript for {uid} UID. Generating transcripts for {uid}")
        print(f"{uid} UID is NOT present in tPrimaryInfo.")
    if uid:
        transmon_response = fetch_api_result(uid)
        gemini_response = generate_output_brcp(uid, created_on)
        status = {"TransmonResponse": transmon_response, "GeminiResponse": gemini_response}
        reportStatus(status)
    else:
        status = {"status": "Fetching latest Upload ID Failed", "message": "Upload Id not found"}
        reportError(status)

    return status


@app.post("/brcp/generateAnalyse/{uid}")
def get_brcp_result_generate_analyse_uid(uid):
    print("started")
    created_on = get_created_on_by_uid(INPUT_DATABASE, uid)
    print("created on done")
    reportStatus(f"Generating and Analysing for UID {uid} created on {created_on}")
    if uid:
        transmon_response = fetch_api_result(uid)
        gemini_response = generate_output_brcp(uid, created_on)
        status = {"TransmonResponse": transmon_response, "GeminiResponse": gemini_response}
        reportStatus(status)
    else:
        status = {"status": "Fetching latest Upload ID Failed", "message": "Upload Id not found"}
        reportError(status)
    return status


@app.post("/brcp/analyse/{uid}")
def get_brcp_result_analyse(uid):
    created_on = get_created_on_by_uid(INPUT_DATABASE, uid)
    reportStatus(f"Analysing for UID {uid} created on {created_on}")
    if uid:
        gemini_response = generate_output_brcp(uid, created_on)
        status = {"TransmonResponse": "Already in DB", "GeminiResponse": gemini_response}
        reportStatus(status)
    else:
        status = {"status": "Fetching latest Upload ID Failed", "message": "Upload Id not found"}
        reportError(status)
    return status


@app.post("/brcp/generate/{uid}")
def get_brcp_result_generate(uid):
    created_on = get_created_on_by_uid(INPUT_DATABASE, uid)
    reportStatus(f"Only Generating transcripts for UID {uid} created on {created_on}")
    if uid:
        transmon_response = fetch_api_result(uid)
        status = {"TransmonResponse": transmon_response}
        reportStatus(status)
    else:
        status = {"status": "Fetching latest Upload ID Failed", "message": "Upload Id not found"}
        reportError(status)
    return status


def generate_output_softskill(date: str):
    responseSoftSkill = {}
    try:
        # Fetch data
        primaryInfo_df, transcript_df, transcriptChat_df, responseDB = fetch_data_softskill(date)
        responseSoftSkill['responseDB'] = responseDB

        # Function to check if a DataFrame is invalid
        def is_invalid_df(df):
            return df is None or df.empty or (len(df) == 1 and df.columns.tolist() == df.iloc[0].tolist())

        # Validate data
        if is_invalid_df(primaryInfo_df) or is_invalid_df(transcript_df) or is_invalid_df(transcriptChat_df):
            error = "❌ One or more dataframes are empty, None, or contain only a header row. Cannot proceed."
            reportError(error)
            responseSoftSkill["FetchError"] = error

        # Analyze data
        analysisResponse = analyse_data_for_soft_skill(primaryInfo_df, transcript_df, transcriptChat_df, date)
        responseSoftSkill["AnalysisResponse"] = analysisResponse

    except Exception as e:
        reportError(f"❌ Error in generate_output_softskill: {str(e)}")
        responseSoftSkill["ExceptionOccurred"] = e
    return responseSoftSkill


@app.get("/softskill")
def get_softskill_result():
    ist = pytz.timezone('Asia/Kolkata')
    date = (datetime.now(ist) - timedelta(days=1)).date()
    print("req date in IST:", date)
    reportStatus(f"Starting Softskill Parameter for {date}")
    softskill_response = generate_output_softskill(date)
    reportStatus(softskill_response)

    return {"database response": softskill_response}


@app.get("/softskill/analyse/{date}")
def get_softskill_result_by_date(date):
    print("req date in IST:", date)
    reportStatus(f"Starting Softskill Parameter for {date}")
    softskill_response = generate_output_softskill(date)
    reportStatus(softskill_response)

    return {"database response": softskill_response}


@app.get('/opsguru')
def getOpsguruResult():
    response = {}
    # Get yesterday's date in Indian format (DD-MM-YYYY)
    yesterday = datetime.now() - timedelta(days=1)
    # Convert explicitly to Year-Month-Day (YYYY-MM-DD) format
    yesterday_ymd = yesterday.strftime('%Y-%m-%d')

    softskill, softskillResponse = fetchSoftskillOpsguru(yesterday_ymd)
    response['softskill'] = softskillResponse
    brcp, BrcpResponse = fetchBrcpOpsguru(yesterday_ymd)
    brcp = brcp.drop(['TL_Email_Id', 'Location'], axis=1)

    response['brcp'] = BrcpResponse
    interaction, interactionResponse = fetchInteractionOpsguru(yesterday_ymd)
    response['interaction'] = interactionResponse
    roster, rosterResponse = fetchRoster()
    response['roster'] = rosterResponse

    if all([df is not None and not df.empty for df in [softskill, brcp, interaction, roster]]):
        print("All DataFrames have data!")
        OpsGuru_df = createDfOpsguru(softskill, brcp, interaction, roster, yesterday_ymd)
        uploadOpsgurudata(OpsGuru_df, table_name='OpsGuruDB')
        reportStatus("OpsGuru Data Uploaded Successfully!")
    else:
        reportError("OpsGuru: Some DataFrames are empty or None!")


@app.get('/opsguru/analyse/{date}')
def getOpsguruResultByDate(date):
    response = {}
    print("req date in IST:", date)

    softskill, softskillResponse = fetchSoftskillOpsguru(date)
    response['softskill'] = softskillResponse
    brcp, BrcpResponse = fetchBrcpOpsguru(date)
    brcp = brcp.drop(['TL_Email_Id', 'Location'], axis=1)

    response['brcp'] = BrcpResponse
    interaction, interactionResponse = fetchInteractionOpsguru(date)
    response['interaction'] = interactionResponse
    roster, rosterResponse = fetchRoster()
    response['roster'] = rosterResponse

    if all([df is not None and not df.empty for df in [softskill, brcp, interaction, roster]]):
        print("All DataFrames have data!")
        OpsGuru_df = createDfOpsguru(softskill, brcp, interaction, roster, date)
        uploadOpsgurudata(OpsGuru_df, table_name='OpsGuruDB')
        reportStatus("OpsGuru Data Uploaded Successfully!")
    else:
        reportError("OpsGuru: Some DataFrames are empty or None!")