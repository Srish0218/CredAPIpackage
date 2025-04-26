from datetime import datetime
import pandas as pd
import pytz
from spacy.language import Language
from spacy_langdetect import LanguageDetector
from ZulipMessenger import reportError, reportStatus
from fetchData import upload_softskill_result_on_database
from parameters import updating_RudeSarcasm_result, classify_rude_sarcastic, \
    process_transcripts_escalation, classify_supervisor, classify_langSwitch, classifyApologyEmpathy, \
    classifyUnethicalSolicitation, classifyReassurance, classifyChatClosing, classifyChatOpening, \
    create_final_DSAT_results, classify_DSAT, classifyVoiceOfCustomer, classifyOpeningLang, processing_timely_closing, \
    calculate_row_language_percentage_spacy, classifyPersonalization, process_TimelyOpening, process_classification, \
    process_hold_data, apply_hold_logic, process_dead_air, merge_hold_and_dead_air, aggregate_dead_air_data, \
    categorize_hold_status
from resources.RefiningResults import merge_all_dataframes, main_processing_pipeline
from resources.working_with_files import merge_dataframes, validate_SOFTSKILL_dataframe, \
    REQUIRED_COLUMNS_SOFTSKILL, validate_brcp_dataframe, REQUIRED_COLUMNS_BRCP


def analyse_data_using_gemini_for_brcp(df, uid, date):
    # Step 1: Sarcasm & Rudeness Classification
    rude_columns = ['Sarcasm_rude_behaviour', 'Sarcasm_rude_behaviour_evidence']
    RudeSarcastic_res_df = process_classification(classify_rude_sarcastic, df, rude_columns, "Rude and Sarcastic")
    # Apply result updates
    RudeSarcastic_res_df = RudeSarcastic_res_df.apply(updating_RudeSarcasm_result, axis=1)

    # Step 2: Escalation Processing
    escalation_columns = [
        'escalation_results', 'Issue_Identification', 'Probable_Reason_for_Escalation',
        'Probable_Reason_for_Escalation_Evidence', 'Agent_Handling_Capability', "Escalation_Category"
    ]
    escalation_res_df = process_classification(process_transcripts_escalation, df, escalation_columns, "Escalation")

    # Step 3: Supervisor Classification
    supervisor_columns = [
        'Wanted_to_connect_with_supervisor', 'de_escalate', 'Supervisor_call_connected',
        'call_back_arranged_from_supervisor', 'supervisor_evidence',
        'Denied_for_Supervisor_call', 'denied_evidence'
    ]
    supervisor_res_df = process_classification(classify_supervisor, df, supervisor_columns, "Supervisor Connect")

    CRED_FINAL_OUTPUT = df[['conversation_id', 'request_id']]
    for df, name in zip([RudeSarcastic_res_df, escalation_res_df, supervisor_res_df],
                        ['RudeSarcastic', 'Escalation', 'Supervisor']):
        CRED_FINAL_OUTPUT = merge_dataframes(CRED_FINAL_OUTPUT, df)

    CRED_FINAL_OUTPUT.replace('nan', 'N/A', inplace=True)

    CRED_FINAL_OUTPUT['Today_Date'] = date.split()[0]
    for index, row in CRED_FINAL_OUTPUT.iterrows():
        if row['Wanted_to_connect_with_supervisor'] == "No":
            for col in ['de_escalate', 'Supervisor_call_connected', 'call_back_arranged_from_supervisor',
                        'supervisor_evidence', 'Denied_for_Supervisor_call']:
                CRED_FINAL_OUTPUT.at[index, col] = "N/A"
        if row['escalation_results'] == "Met":
            for col in ['Issue_Identification', 'Probable_Reason_for_Escalation',
                        'Probable_Reason_for_Escalation_Evidence', 'Agent_Handling_Capability', 'Escalation_Category']:
                CRED_FINAL_OUTPUT.at[index, col] = "N/A"
        if row['Sarcasm_rude_behaviour'] == "Met":
            CRED_FINAL_OUTPUT.at[
                index, 'Sarcasm_rude_behaviour_evidence'] = "The agent remained polite and professional."

    CRED_FINAL_OUTPUT.fillna("N/A", inplace=True)
    try:
        CRED_FINAL_OUTPUT['uploaded_id'] = str(uid)
    except Exception as e:
        print(e)
    is_valid, missing_cols, extra_cols = validate_brcp_dataframe(CRED_FINAL_OUTPUT)

    if is_valid:
        CRED_FINAL_OUTPUT = CRED_FINAL_OUTPUT[REQUIRED_COLUMNS_BRCP]
        return CRED_FINAL_OUTPUT
    else:
        if missing_cols:
            body = "❌ The Output Data is missing the following columns:\n\n" + ", ".join(missing_cols)
            reportError(body)
            return None

        if extra_cols:
            body = "⚠️ The Output Data contains extra columns not in the required list:\n\n" + ", ".join(
                extra_cols) + "\nExtra Columns are dropped."
            CRED_FINAL_OUTPUT = CRED_FINAL_OUTPUT.drop(columns=extra_cols, errors="ignore")
            reportError(body)
            return CRED_FINAL_OUTPUT


def analyse_data_for_soft_skill(primaryInfo_df, transcript_df, transcriptChat_df, date):
    try:
        primaryInfo_df = primaryInfo_df[['conversation_id', 'request_id', 'Time_duration_of_Call', 'surveypoint',
                                         'Total_instance_long_dead_Air', 'Total_instance_short_dead_Air',
                                         'totalholdtime', 'calldisconnectionby']]
    except KeyError as e:
        keyError = f"KeyError: Missing column {e} in primaryInfo_df"
        print(keyError)
        reportError(keyError)
    except Exception as e:
        error = f"Unexpected error while filtering primaryInfo_df: {e}"
        print(error)
        reportError(error)

    try:
        transcriptChat_df = transcriptChat_df.sort_values(by=['request_id', 'id'])
    except KeyError as e:
        keyError = f"KeyError: Missing column {e} in transcriptChat_df"
        print(keyError)
        reportError(keyError)
    except Exception as e:
        error = f"Unexpected error while sorting transcriptChat_df: {e}"
        print(error)
        reportError(error)

    try:
        transcript_df = pd.merge(transcript_df, primaryInfo_df, how='inner')
    except KeyError as e:
        keyError = f"KeyError: Missing column {e} during transcript_df merge"
        print(keyError)
        reportError(keyError)
    except Exception as e:
        error = f"Unexpected error while merging transcript_df: {e}"
        print(error)
        reportError(error)

    try:
        transcriptChat_df = pd.merge(transcriptChat_df, primaryInfo_df, how='inner')
        transcriptChat_df = transcriptChat_df.sort_values(by=['request_id', 'id'])
    except KeyError as e:
        keyError = f"KeyError: Missing column {e} during transcriptChat_df merge"
        print(keyError)
        reportError(keyError)
    except Exception as e:
        error = f"Unexpected error while merging transcriptChat_df: {e}"
        print(error)
        reportError(error)

    try:
        transcript_ids = transcript_df['request_id'].unique()
        transcriptChat_df = transcriptChat_df[transcriptChat_df['request_id'].isin(transcript_ids)]
    except KeyError as e:
        keyError = f"KeyError: Missing column {e} in transcript_df or transcriptChat_df"
        print(keyError)
        reportError(keyError)
    except Exception as e:
        error = f"Unexpected error while filtering transcriptChat_df: {e}"
        print(error)
        reportError(error)

    try:
        transcript_df.replace("entry", "", inplace=True)
        transcriptChat_df.replace("entry", "", inplace=True)
    except Exception as e:
        error = f"Unexpected error while replacing values: {e}"
        print(error)
        reportError(error)

    try:
        transcript_df = transcript_df.dropna(subset=['transcript'])
        transcriptChat_df = transcriptChat_df.dropna(subset=['transcript'])
    except KeyError as e:
        keyError = f"KeyError: Missing column {e} while dropping NaNs"
        print(keyError)
        reportError(keyError)
    except Exception as e:
        error = f"Unexpected error while dropping NaNs: {e}"
        print(error)
        reportError(error)

    # Step 1: Language Switch Parameter
    # reportStatus(f"Processing Language Switch Parameter...")
    langSwitch_df = classify_langSwitch(transcriptChat_df)
    reportStatus(f"✅ Language Switch Parameter processing complete")

    # Step 2: Empathy and Apology
    empathy_columns = ['Apology_result', 'Apology_evidence', 'Empathy_result', 'Empathy_evidence',
                       'Apology_Category', 'Empathy_Category']

    Empathy_apology_res_df = process_classification(classifyApologyEmpathy, transcript_df, empathy_columns,
                                                    "Apology and Empathy")
    # Step 3: Unethical Solicitation
    unethical_columns = ['Unethical_Solicitation', 'Unethical_Solicitation_Evidence']
    Unethical_Solicitation_res_df = process_classification(classifyUnethicalSolicitation, transcript_df,
                                                           unethical_columns, "Unethical Solicitaion")

    # Step 4: Reassurance Parameter
    Reassurance_columns = ['Reassurance_result', 'Reassurance_evidence', 'Reassurance_Category']
    Reassurance_res_df = process_classification(classifyReassurance, transcript_df, Reassurance_columns, "Reassurance")

    # Step 5: Call Closing Parameter
    ChatClosing_columns = ["Further Assistance", "Further Assistance Evidence", "Effective IVR Survey",
                           "Effective IVR Survey Evidence", "Branding", "Branding Evidence", "Greeting",
                           "Greeting Evidence"]
    ChatClosing_res_df = process_classification(classifyChatClosing, transcript_df, ChatClosing_columns, "Chat Closing")

    # Step 6: Call Opening Parameter
    ChatOpening_columns = ["Greeting_the_customer", "Greeting_the_customer_evidence", "Self_introduction",
                           "Self_introduction_evidence", "Identity_confirmation", "Identity_confirmation_evidence"]
    ChatOpening_res_df = process_classification(classifyChatOpening, transcript_df, ChatOpening_columns, "Chat Opening")


    # Step 7: Survey Pitch Parameter
    Survey_res_df = ChatClosing_res_df[['request_id', "Effective IVR Survey", "Effective IVR Survey Evidence"]].rename(
        columns={'Effective IVR Survey': 'No_Survey_Pitch',
                 'Effective IVR Survey Evidence': 'No_Survey_Pitch_Evidence'})

    # Step 8: DSAT Parameter
    # Convert 'surveypoint' to numeric and fill NaN with 0
    transcript_df["surveypoint"] = pd.to_numeric(transcript_df["surveypoint"], errors='coerce').fillna(0)

    # Filter rows where 'Survey Results' column is less than or equal to 3
    DSAT_df = transcript_df[(transcript_df["surveypoint"] > 0) & (transcript_df["surveypoint"] <= 3)]
    Survey_IDS = DSAT_df['request_id'].astype(str)  # Ensure 'request_id' consistency

    if DSAT_df.empty:
        print("⚠️ No DSAT cases found. Skipping DSAT processing...")
        DSAT_res_df = pd.DataFrame(columns=['request_id', 'Customer_Issue_Identification', 'Reason_for_DSAT',
                                            'Suggestion_for_DSAT_Prevention'])
    else:
        print("🚀 Starting DSAT processing...")
        DSAT_columns = ['Customer_Issue_Identification', 'Reason_for_DSAT', 'Suggestion_for_DSAT_Prevention']
        DSAT_res_df = process_classification(classify_DSAT, DSAT_df, DSAT_columns, "DSAT")

    # Create final DSAT results
    final_DSAT_res_df = create_final_DSAT_results(transcript_df, DSAT_res_df, Survey_IDS)

    # Step 9: Voice Of Customer Parameter
    voice_of_customer_columns = ['VOC_Category', 'VOC_Core_Issue_Summary']
    voice_of_customer_res_df = process_classification(classifyVoiceOfCustomer, transcript_df, voice_of_customer_columns,
                                                      "Voice Of Customer")

    # Convert request_id to string for proper mapping
    voice_of_customer_res_df['request_id'] = voice_of_customer_res_df['request_id'].astype(str)

    # Default DSAT Category as 'N/A'
    voice_of_customer_res_df['DSAT_Category'] = 'N/A'

    # Create mapping dictionary
    voc_category_dict = voice_of_customer_res_df.set_index('request_id')['VOC_Category'].to_dict()

    # Update DSAT_Category for matching request_ids
    voice_of_customer_res_df.loc[voice_of_customer_res_df['request_id'].isin(Survey_IDS), 'DSAT_Category'] = \
        voice_of_customer_res_df['request_id'].apply(lambda x: voc_category_dict.get(x, 'N/A'))

    print("✅ DSAT & VOC Processing Done!")

    # Step 10: Opening Language Parameter
    opening_lang_columns = ['Open the call in default language', 'Open the call in default language evidence',
                            'Open the call in default language Reason']
    opening_lang_res_df = process_classification(classifyOpeningLang, transcript_df, opening_lang_columns,
                                                 "Open the call in default language")

    # Step 9: Timely CLosing Parameter
    # reportStatus(f"Processing Timely CLosing Parameter...")
    timely_closing_res_df = processing_timely_closing(primaryInfo_df, transcript_df, transcriptChat_df, "surveypoint")
    reportStatus(f"✅ Timely CLosing Parameter processing complete")

    print("✅ Timely Closing Done!")

    # Step 10: Hold and dead air Parameter
    # reportStatus(f"Processing Hold and dead air Parameter...")
    final_hold_df = process_hold_data(transcriptChat_df)
    final_hold_df = apply_hold_logic(final_hold_df)
    dead_air_df = process_dead_air(primaryInfo_df, transcriptChat_df)
    dead_air_df = aggregate_dead_air_data(dead_air_df)
    final_hold_df = merge_hold_and_dead_air(final_hold_df, dead_air_df)
    final_hold_df = categorize_hold_status(final_hold_df)
    print("Hold parameter processed...")
    reportStatus(f"✅ Hold and dead air Parameter processing complete")

    # Step 11: Conversation Language Parameter
    # reportStatus(f"Processing Conversation Language Parameter...")
    if not Language.has_factory("language_detector"):
        @Language.factory("language_detector")
        def create_language_detector(nlp, name):
            return LanguageDetector()
    ConversationLang_df = calculate_row_language_percentage_spacy(transcript_df)
    reportStatus(f"✅ Conversation Language Parameter processing complete")

    # Step 12: Personalization Parameter
    Personalization_columns = ['Personalization_result', 'Personalization_Evidence']
    Personalization_res_df = process_classification(classifyPersonalization, transcript_df, Personalization_columns,
                                                    "Personalization")
    print("personalization done")
    reportStatus(f"✅ Personalization Parameter processing complete")

    # Step 13: Timely Opening Parameter
    # reportStatus(f"Processing Timely Opening Parameter...")
    timelyOpening_df = process_TimelyOpening(transcriptChat_df)
    print("timely opening done")
    reportStatus(f"✅ Timely Opening Parameter processing complete")

    print("combing")
    CRED_FINAL_OUTPUT = langSwitch_df

    for df, name in zip(
            [Reassurance_res_df, Empathy_apology_res_df, ChatOpening_res_df, ChatClosing_res_df, Survey_res_df,
             Unethical_Solicitation_res_df,
             final_DSAT_res_df, voice_of_customer_res_df, opening_lang_res_df, timely_closing_res_df, final_hold_df,
             ConversationLang_df, Personalization_res_df,
             timelyOpening_df],
            ['Reassure', 'Apology_And_Empathy', 'Opening', 'Closing', 'Survey', 'Unethical',
             'DSAT', 'voice_of_customer', 'opening_lang', 'timely_closing', 'Hold_parameter', 'Lang_detect',
             'Personalization', 'timelyOpening']):
        CRED_FINAL_OUTPUT = merge_all_dataframes(CRED_FINAL_OUTPUT, df, df2_name=name)

    if CRED_FINAL_OUTPUT.empty:
        dataState = "The final merged Data is empty. No data to save."
        print(dataState)
        reportError(dataState)
        return dataState

    CRED_FINAL_OUTPUT = main_processing_pipeline(CRED_FINAL_OUTPUT, primaryInfo_df)
    if CRED_FINAL_OUTPUT is None:
        dataStatus = f"Final DataFrame is empty. No data to save."
        reportError(dataStatus)
        return dataStatus
    CRED_FINAL_OUTPUT = CRED_FINAL_OUTPUT[REQUIRED_COLUMNS_SOFTSKILL]

    is_valid, missing_cols, extra_cols = validate_SOFTSKILL_dataframe(CRED_FINAL_OUTPUT)

    if is_valid:
        CRED_FINAL_OUTPUT["uploaded_date"] = date
        print(CRED_FINAL_OUTPUT)
        reportStatus(f"✅ CRED Final Output is merged and validated")
        response = upload_softskill_result_on_database(CRED_FINAL_OUTPUT, date)
        return response
    else:
        if missing_cols:
            body = "❌ The Output Data is missing the following columns:\n\n" + ", ".join(missing_cols)
            reportError(body)
            return None

        if extra_cols:
            body = "⚠️ The Output Data contains extra columns not in the required list:\n\n" + ", ".join(
                extra_cols) + "\nExtra Columns are dropped."
            CRED_FINAL_OUTPUT = CRED_FINAL_OUTPUT.drop(columns=extra_cols, errors="ignore")
            reportStatus(body)
            return CRED_FINAL_OUTPUT
