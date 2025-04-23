from datetime import datetime, timedelta

import pandas as pd
import pytz

from ZulipMessenger import reportError


def merge_dataframes(df1, df2):
    """Merge two dataframes on a given column, handling errors gracefully"""
    if df2.empty:
        return df1

    try:
        return df1.merge(df2, on="request_id", how='inner')
    except KeyError as e:
        reportError(f"KeyError: Column request_id not found in one of the DataFrames. Returning df1.")
        return df1
    except Exception as e:
        reportError(f"Error merging dataframes: {e}. Returning df1.")
        return df1


# Define required columns
REQUIRED_COLUMNS_BRCP = [
    "conversation_id", "request_id", "Sarcasm_rude_behaviour", "Sarcasm_rude_behaviour_evidence",
    "escalation_results", "Issue_Identification", "Probable_Reason_for_Escalation",
    "Probable_Reason_for_Escalation_Evidence", "Agent_Handling_Capability",
    "Wanted_to_connect_with_supervisor", "de_escalate", "Supervisor_call_connected",
    "call_back_arranged_from_supervisor", "supervisor_evidence", "Denied_for_Supervisor_call",
    "denied_evidence", "Today_Date", "uploaded_id", "Escalation_Category"
]


def validateDataframes(df, columns):
    """Check if the DataFrame contains exactly the required columns."""
    df_columns = set(df.columns)
    required_set = set(columns)

    missing_columns = list(required_set - df_columns)  # Columns that are required but missing
    extra_columns = list(df_columns - required_set)  # Columns that should not be there

    if not missing_columns and not extra_columns:
        return True, [], []
    return False, missing_columns, extra_columns


def validate_brcp_dataframe(df):
    """Check if the DataFrame contains exactly the required columns."""
    df_columns = set(df.columns)
    required_set = set(REQUIRED_COLUMNS_BRCP)

    missing_columns = list(required_set - df_columns)  # Columns that are required but missing
    extra_columns = list(df_columns - required_set)  # Columns that should not be there

    if not missing_columns and not extra_columns:
        return True, [], []
    return False, missing_columns, extra_columns


REQUIRED_COLUMNS_SOFTSKILL = ['conversation_id', 'request_id', 'hold_request_found', 'hold_evidence',
                              'CustomerLangCount', 'AgentLangCount',
                              'language_switch',
                              'Reassurance_result', 'Reassurance_evidence', 'Apology_result',
                              'Apology_evidence', 'Empathy_result', 'Empathy_evidence',
                              'No_Survey_Pitch', 'No_Survey_Pitch_Evidence', 'Unethical_Solicitation',
                              'Unethical_Solicitation_Evidence',
                              'DSAT_result', 'Customer_Issue_Identification', 'Reason_for_DSAT',
                              'Suggestion_for_DSAT_Prevention', 'DSAT_Category', 'Open the call in default language',
                              'Open the call in default language evidence', 'Open the call in default language Reason',
                              'Hold_requested_before_dead_air', 'long_dead_air', 'dead_air_timestamp',
                              'VOC_Category', 'VOC_Core_Issue_Summary', 'timely_closing_result',
                              'timely_closing_evidence', 'hold_ended_in_required_duration',
                              'hold_ended_in_required_duration_evidence', 'hold_durations_after_hold_request',
                              'language_switch_result', 'Call_Opening_Category', 'default_opening_lang_Category',
                              'Apology_Category', 'Empathy_Category', 'Chat_Closing_Category',
                              'language_switch_category',
                              'Hold_category', 'Reassurance_Category', 'Language', 'Personalization_result',
                              'Personalization_Evidence', 'Delayed call opening', 'Delayed call opening evidence',
                              "Further Assistance", "Further Assistance Evidence", "Effective IVR Survey",
                              "Effective IVR Survey Evidence", "Branding", "Branding Evidence", "Greeting",
                              "Greeting Evidence",
                              "Greeting_the_customer", "Greeting_the_customer_evidence", "Self_introduction",
                              "Self_introduction_evidence",
                              "Identity_confirmation", "Identity_confirmation_evidence"]


def validate_SOFTSKILL_dataframe(df):
    """Check if the DataFrame contains exactly the required columns."""
    df_columns = set(df.columns)
    required_set = set(REQUIRED_COLUMNS_SOFTSKILL)

    missing_columns = list(required_set - df_columns)  # Columns that are required but missing
    extra_columns = list(df_columns - required_set)  # Columns that should not be there

    if not missing_columns and not extra_columns:
        return True, [], []
    return False, missing_columns, extra_columns


def categorize_missing_columns(missing_cols):
    """Categorize missing columns into Sarcasm, Escalation, or Supervisor related."""
    sarcasm_cols = {'Sarcasm_rude_behaviour', 'Sarcasm_rude_behaviour_evidence'}
    escalation_cols = {'escalation_results', 'Issue_Identification', 'Probable_Reason_for_Escalation',
                       'Probable_Reason_for_Escalation_Evidence', 'Agent_Handling_Capability'}
    supervisor_cols = {'Wanted_to_connect_with_supervisor', 'de_escalate', 'Supervisor_call_connected',
                       'call_back_arranged_from_supervisor', 'supervisor_evidence', 'Denied_for_Supervisor_call',
                       'denied_evidence'}

    categories = []
    if any(col in sarcasm_cols for col in missing_cols):
        categories.append("Sarcasm")
    if any(col in escalation_cols for col in missing_cols):
        categories.append("Escalation")
    if any(col in supervisor_cols for col in missing_cols):
        categories.append("Supervisor")

    return categories

def get_time():
    """Get the current IST time formatted as '01_April_2025_558PM'."""
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist) - timedelta(hours=1)
    return current_time.strftime("%d_%B_%Y_%I%M%p")


def calculate_aht(duration):
    try:
        if pd.isna(duration):
            return "Unknown"
        elif duration < 120:
            return "Less than 2 min"
        elif duration <= 300:
            return "2 to 5 min"
        elif duration <= 600:
            return "5 to 10 min"
        return "Above 10 min"
    except Exception as e:
        print(f"Error in AHT calculation: {e}")
        return "Error"
def createDfOpsguru(soft_skill_df, brcp_df, interaction_df, roaster_df, date):
    if brcp_df is not None and not brcp_df.empty:
        brcp_df = brcp_df.drop(columns=["request_id", "Today_Date"], errors="ignore")
        soft_skill_brcp_df = pd.merge(soft_skill_df, brcp_df, on="conversation_id", how="inner")

    soft_skill_brcp_interaction_df = pd.merge(soft_skill_brcp_df, interaction_df,
                                              left_on="conversation_id",
                                              right_on="conversationid", how="inner")
    rename_dict = {
        "mediatype": "Source",
        "campaignname": "Group Type",
        "startdatetime": "Resolved Time",
        "surveypoint": "Survey Result",
        "agentemail1": "Agentid",
        "queuename1": "Vertical",
        "wrapupnamefirst": "Wrapupcode",
        "freshdeskticketid": "Ticket id"
    }
    soft_skill_brcp_interaction_df.rename(columns=rename_dict, inplace=True)
    soft_skill_brcp_interaction_df['Status'] = "-"
    soft_skill_brcp_interaction_df['Tags'] = "-"
    soft_skill_brcp_interaction_df['Supervisor Call Type'] = "-"
    soft_skill_brcp_interaction_df['Duration sec'] = soft_skill_brcp_interaction_df['duration'] / 1000
    soft_skill_brcp_interaction_df["AHT Bucket"] = soft_skill_brcp_interaction_df['Duration sec'].apply(
        calculate_aht)
    soft_skill_brcp_interaction_roaster_df = pd.merge(soft_skill_brcp_interaction_df, roaster_df,
                                                      left_on="Agentid",
                                                      right_on="Email_Id", how="left")
    soft_skill_brcp_interaction_roaster_df.rename(
        columns={"Location": "Center", "TL_Email_Id": "TL ID"},
        inplace=True)
    soft_skill_brcp_interaction_roaster_df.to_excel("soft_skill_brcp_interaction_roaster_df.xlsx")
    final_columns = [
        'Center', 'TL ID', 'Group Type', 'Resolved Time', 'Source', 'Status',
        'Survey Result', 'Tags', 'Agentid', 'AHT Bucket', 'Vertical',
        'Supervisor Call Type', 'Wrapupcode', 'Ticket id', 'Language',
        'conversation_id', 'request_id', 'hold_request_found', 'hold_evidence',
        'CustomerLangCount', 'AgentLangCount', 'language_switch',
        'Sarcasm_rude_behaviour', 'Sarcasm_rude_behaviour_evidence',
        'Reassurance_result', 'Reassurance_evidence', 'Apology_result',
        'Apology_evidence', 'Empathy_result', 'Empathy_evidence',
        'escalation_results', 'Issue_Identification',
        'Probable_Reason_for_Escalation',
        'Probable_Reason_for_Escalation_Evidence', 'Agent_Handling_Capability',
        'No_Survey_Pitch', 'No_Survey_Pitch_Evidence', 'Unethical_Solicitation',
        'Unethical_Solicitation_Evidence', 'Wanted_to_connect_with_supervisor',
        'de_escalate', 'Supervisor_call_connected',
        'call_back_arranged_from_supervisor', 'supervisor_evidence',
        'DSAT_result', 'Customer_Issue_Identification', 'Reason_for_DSAT',
        'Suggestion_for_DSAT_Prevention', 'Denied_for_Supervisor_call',
        'denied_evidence', 'DSAT_Category', 'Open_the_call_in_default_language',
        'Open_the_call_in_default_language_evidence',
        'Open_the_call_in_default_language_Reason',
        'Hold_requested_before_dead_air', 'long_dead_air', 'dead_air_timestamp',
        'VOC_Category', 'VOC_Core_Issue_Summary', 'timely_closing_result',
        'timely_closing_evidence', 'hold_ended_in_required_duration',
        'hold_ended_in_required_duration_evidence',
        'hold_durations_after_hold_request', 'language_switch_result',
        'Call_Opening_Category', 'default_opening_lang_Category',
        'Apology_Category', 'Empathy_Category', 'Chat_Closing_Category',
        'language_switch_category', 'Hold_category', 'Reassurance_Category',
        'Personalization_result', 'Personalization_Evidence',
        'Delayed_call_opening', 'Delayed_call_opening_evidence',
        'Further_Assistance', 'Further_Assistance_Evidence',
        'Effective_IVR_Survey', 'Effective_IVR_Survey_Evidence', 'Branding',
        'Branding_Evidence', 'Greeting', 'Greeting_Evidence',
        'Greeting_the_customer', 'Greeting_the_customer_evidence',
        'Self_introduction', 'Self_introduction_evidence',
        'Identity_confirmation', 'Identity_confirmation_evidence', 'Escalation_Category'
    ]
    soft_skill_brcp_interaction_roaster_df = soft_skill_brcp_interaction_roaster_df[final_columns]
    soft_skill_brcp_interaction_roaster_df.fillna("N/A", inplace=True)
    soft_skill_brcp_interaction_roaster_df.drop_duplicates()
    soft_skill_brcp_interaction_roaster_df['Uploaded_Date'] = date
    return soft_skill_brcp_interaction_roaster_df

