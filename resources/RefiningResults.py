import re


# Updating functions
def updating_CRED_FINAL_OUTPUT_results(main_df):
    # Loop for 'unethical solicitation'
    for index, r in main_df.iterrows():
        if r['Unethical_Solicitation'] == "Not Met":
            evidence = r['Unethical_Solicitation_Evidence']
            non_unethical_pattern = r'\b(not explicitly ask for a high rating|not constitute an unethical solicitation)\b'
            non_unethical_found = bool(re.search(non_unethical_pattern, evidence, re.IGNORECASE))

            if non_unethical_found:
                main_df.at[index, 'Unethical_Solicitation'] = "Not Met"

    # Loop for 'No_Survey_Pitch'
    for index, r in main_df.iterrows():
        if r['No_Survey_Pitch'] == "Not Met":
            main_df.at[index, 'Unethical_Solicitation'] = None
            main_df.at[index, 'Unethical_Solicitation_Evidence'] = None

    # Loop for 'default_opening_lang'
    for index, r in main_df.iterrows():
        if r['Open the call in default language evidence']:
            evidence = r['Open the call in default language evidence']

            # 1. Check for greeting
            greeting_pattern = r'\b(Good morning|Good afternoon|Good evening|good|hi|Hello)\b'
            greeting_found = bool(re.search(greeting_pattern, evidence, re.IGNORECASE))

            # 2. Check for self-introduction
            introduction_pattern = r'\b(This is|My name is|this side|Myself|calling from|I\'m|it\'s)\b'
            introduction_found = bool(re.search(introduction_pattern, evidence, re.IGNORECASE))

            # 3. Check for confirming customer's name
            customer_name_pattern = r'\b(Is this|Am I speaking|Am I talking)\b'
            customer_name_confirmation_found = bool(re.search(customer_name_pattern, evidence, re.IGNORECASE))

            # Count false checks
            false_checks = sum([not greeting_found, not introduction_found, not customer_name_confirmation_found])

            # Assign value
            main_df.at[index, 'Open the call in default language'] = "Not Met" if false_checks > 2 else "Met"

            # Set 'opening_lang_Reason'
            if main_df.at[index, 'Open the call in default language'] == "Met":
                main_df.at[index, 'Open the call in default language Reason'] = (
                    "The agent greeted the customer in English, introduced themselves in English, and confirmed the "
                    "customer's name."
                )

    # Add 'language_switch_result' column
    main_df['language_switch_result'] = "Met"

    # Loop for 'language switch'
    for index, r in main_df.iterrows():
        if r['language_switch'] == "Customer spoke in Hindi but agent didn't switch language":
            main_df.at[index, 'language_switch_result'] = 'Not Met'

    return main_df


def addingCategories(main_df):
    # Categorize Call Opening
    for index, row in main_df.iterrows():
        missing_categories = []

        # Check for Greeting
        if row['Greeting_the_customer'] == "Not Met":
            missing_categories.append("GreetingMissing")

        # Check for Self-Introduction
        if row['Self_introduction'] == "Not Met":
            missing_categories.append("SelfIntroductionMissing")

        # Check for Identity Confirmation
        if row['Identity_confirmation'] == "Not Met":
            missing_categories.append("NameConfirmationMissing")

        # Assign category
        main_df.at[index, 'Call_Opening_Category'] = ", ".join(
            missing_categories) if missing_categories else "Guidelines Followed."

    # Categorize Default Opening Language
    for index, row in main_df.iterrows():
        if main_df.at[index, 'Open the call in default language'] == "Not Met":
            main_df.at[index, 'default_opening_lang_Category'] = "Failed to open the call in default language (English)"
        else:
            main_df.at[index, 'default_opening_lang_Category'] = "Guidelines Followed."

    # Categorize Apology & Empathy
    for index, row in main_df.iterrows():
        # Remove "Partially"
        main_df.at[index, 'Apology_result'] = str(main_df.at[index, 'Apology_result']).replace("Partially", "").strip()
        main_df.at[index, 'Empathy_result'] = str(main_df.at[index, 'Empathy_result']).replace("Partially", "").strip()

        # Assign category
        if main_df.at[index, 'Apology_result'] == "Met":
            main_df.at[index, 'Apology_Category'] = "Guidelines Followed."
        if main_df.at[index, 'Empathy_result'] == "Met":
            main_df.at[index, 'Empathy_Category'] = "Guidelines Followed."

    # Categorize Call Closing
    for index, row in main_df.iterrows():
        missing_categories = []

        # Check each condition
        if row['Further Assistance'] == "Not Met":
            missing_categories.append("Further Assistance Missing")

        if row['Effective IVR Survey'] == "Not Met":
            missing_categories.append("Survey Feedback Missing")

        if row['Greeting'] == "Not Met":
            missing_categories.append("Closing Greetings Missing")

        # Assign category
        main_df.at[index, 'Chat_Closing_Category'] = ", ".join(
            missing_categories) if missing_categories else "Guidelines Followed."

    # Reassurance category
    for index, row in main_df.iterrows():
        if main_df.at[index, 'Reassurance_result'] == "Met":
            main_df.at[index, 'Reassurance_Category'] = "Guidelines Followed"

    return main_df


def merge_all_dataframes(df1, df2, on_column='request_id', df2_name=''):
    # Merge two DataFrames on a specific column with inner join.
    if df2.empty:
        print(f"{df2_name} DataFrame is empty. Skipping merge.")
        return df1  # Skip merging if df2 is empty

    try:
        return df1.merge(df2, on=on_column, how='inner')
    except Exception as e1:
        print(f"Error merging with {df2_name} on {on_column}: {e1}")
        return df1  # Return df1 if merging fails


def preprocess_dataframe(df):
    """
    Handle NaN values and format data correctly.
    """
    if df is None:
        print("Final DataFrame is empty. No processing required.")
        return None

    df = df.replace('nan', 'N/A').fillna("N/A")
    return df


def update_closing_values(df, primary_info_df):
    """
    Update 'Not Met' closing values based on call disconnection status.
    """
    if df is None:
        return None

    # Create lookup dictionary for request_id -> call disconnection status
    call_disconnection_dict = dict(zip(primary_info_df['request_id'], primary_info_df['calldisconnectionby']))

    closing_columns = ['Further Assistance', 'Effective IVR Survey', 'Greeting']
    evidence_columns = ['Further Assistance Evidence', 'Effective IVR Survey Evidence', 'Greeting Evidence']

    # Condition to check 'Not Met' and 'Call disconnected by Customer/System'
    condition = df['request_id'].map(call_disconnection_dict).isin(['Customer', 'System'])

    for col, evidence_col in zip(closing_columns, evidence_columns):
        col_condition = (df[col] == 'Not Met') & condition
        df.loc[col_condition, col] = 'Met'

        # Ensure evidence column has string values before concatenation
        df.loc[col_condition, evidence_col] = df.loc[col_condition, evidence_col].astype(str) + ("Call was "
                                                                                                 "disconnected by the"
                                                                                                 " Customer/System, "
                                                                                                 "so marked as Met.")

    print("Closing values updated successfully.")
    return df


def main_processing_pipeline(CRED_FINAL_OUTPUT, primaryInfo_df):
    """
    Main function to merge, process, and update the DataFrame.
    """
    # Preprocess the merged DataFrame
    processed_df = preprocess_dataframe(CRED_FINAL_OUTPUT)
    processed_df = addingCategories(processed_df)
    # Update closing conditions
    final_df = update_closing_values(processed_df, primaryInfo_df)

    final_df = updating_CRED_FINAL_OUTPUT_results(final_df)

    # Ensure DataFrame is not empty before saving
    if final_df is not None and not final_df.empty:
        dataStatus = f"CRED_FINAL_OUTPUT ready to upload"
        print(dataStatus)
        return final_df
    else:
        print(f"Final DataFrame is empty. No data to save.")
        return final_df
