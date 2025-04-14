from datetime import datetime, timedelta
import pandas as pd
import pytz

from fetchData import fetchInteractionRoaster_forBrcp, upload_cred_result_on_database

# Metadata
uid = "ETL_92200"
created_on = "2025-04-14 12:00:10"

# Read input Excel
final_df = pd.read_excel("Final.xlsx")

# Get today's date in IST
ist = pytz.timezone('Asia/Kolkata')
date = (datetime.now(ist) - timedelta(days=0)).date()

# Fetch interaction roster
interaction_roster_df = fetchInteractionRoaster_forBrcp(date)

# Merge dataframes on conversation ID
interaction_roster_brcp_df = final_df.merge(
    interaction_roster_df,
    left_on="conversation_id",
    right_on="conversationid",
    how="left"
)

# Drop Unnamed columns (index columns) if present
interaction_roster_brcp_df = interaction_roster_brcp_df.loc[:, ~interaction_roster_brcp_df.columns.str.contains('^Unnamed')]

# Drop 'conversationid' column if it exists
if 'conversationid' in interaction_roster_brcp_df.columns:
    interaction_roster_brcp_df = interaction_roster_brcp_df.drop(columns=['conversationid', 'agentemail1'])
interaction_roster_brcp_df.to_excel("interaction_roster_brcp_df.xlsx", index=False)

# List of required columns
required_columns = [
    "conversation_id", "request_id", "Sarcasm_rude_behaviour", "Sarcasm_rude_behaviour_evidence",
    "escalation_results", "Issue_Identification", "Probable_Reason_for_Escalation",
    "Probable_Reason_for_Escalation_Evidence", "Agent_Handling_Capability",
    "Wanted_to_connect_with_supervisor", "de_escalate", "Supervisor_call_connected",
    "call_back_arranged_from_supervisor", "supervisor_evidence", "Denied_for_Supervisor_call",
    "denied_evidence", "Today_Date", "uploaded_id", "Escalation_Category", "Location",
    "TL_Email_Id", "Email_ID"
]

# Reorder: Place desired columns first (if they exist), then all other columns
current_cols = interaction_roster_brcp_df.columns.tolist()
ordered_cols = [col for col in required_columns if col in current_cols]  # only existing desired columns
remaining_cols = [col for col in current_cols if col not in ordered_cols]  # the rest
final_column_order = ordered_cols + remaining_cols

# Apply new column order
interaction_roster_brcp_df = interaction_roster_brcp_df[final_column_order]

# Upload to database
msg = upload_cred_result_on_database(interaction_roster_brcp_df, uid, created_on)

# Save to Excel

print(msg)
