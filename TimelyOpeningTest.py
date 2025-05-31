from datetime import datetime, timedelta
import pytz
import pandas as pd
from fetchData import upload_softskill_result_on_database, fetch_data_from_database_by_date

def process_TimelyOpening(dataframe):
    # Convert 'starttime' to numeric, forcing errors to NaN if conversion fails
    dataframe['starttime'] = pd.to_numeric(dataframe['starttime'], errors='coerce')

    # Extract the first occurrence of each unique request_id
    first_rows = dataframe.groupby("request_id").first().reset_index()

    # Ensure 'starttime' is numeric
    first_rows['starttime'] = pd.to_numeric(first_rows['starttime'], errors='coerce')

    # Initialize result columns with default values
    first_rows['Delayed call opening'] = 'Not Met'
    first_rows['Delayed call opening evidence'] = 'Start time is missing.'

    # Apply logic for Voice
    voice_mask = (first_rows['mediatype'] == 'voice') & first_rows['starttime'].notna()
    first_rows.loc[voice_mask, 'Delayed call opening'] = first_rows.loc[voice_mask, 'starttime'].apply(
        lambda x: 'Met' if x <= 3 else 'Not Met'
    )
    first_rows.loc[voice_mask, 'Delayed call opening evidence'] = first_rows.loc[voice_mask, 'starttime'].apply(
        lambda x: f'Call opened within 3 seconds. Agent opened the call at {x}.' if x <= 3
        else f'Call opened after 3 seconds. Agent opened the call at {x}.'
    )

    # Apply logic for Callback
    callback_mask = (first_rows['mediatype'] == 'Callback') & first_rows['starttime'].notna()
    first_rows.loc[callback_mask, 'Delayed call opening'] = first_rows.loc[callback_mask, 'starttime'].apply(
        lambda x: 'Met' if x < 21 else 'Not Met'
    )
    first_rows.loc[callback_mask, 'Delayed call opening evidence'] = first_rows.loc[callback_mask, 'starttime'].apply(
        lambda x: f'Call opened within 21 seconds. Agent opened the call at {x}.' if x < 21
        else f'Call opened after 21 seconds. Agent opened the call at {x}.'
    )

    # Return only the necessary columns
    return first_rows[['request_id', 'mediatype', 'starttime', 'Delayed call opening', 'Delayed call opening evidence']]

ist = pytz.timezone('Asia/Kolkata')
date = (datetime.now(ist) - timedelta(days=1)).date()
print("req date in IST:", date)
df_raw = fetch_data_from_database_by_date(date)

if df_raw is not None and not df_raw.empty:
    timelyOpening_df = process_TimelyOpening(df_raw)
    print("✅ Timely Opening process completed.")
    print(timelyOpening_df.head())
    file_name = f"Timely_Opening_{date}.xlsx"

    # Save data to Excel with multiple sheets
    with pd.ExcelWriter(file_name, engine="xlsxwriter") as writer:
        timelyOpening_df.to_excel(writer, sheet_name="Timely Opening", index=False)

    print(f"Excel file '{file_name}' created successfully with 3 sheets!")


else:
    print("❌ No data fetched for the given date. Timely Opening processing skipped.")
