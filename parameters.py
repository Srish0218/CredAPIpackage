import re
import time

import spacy
import langid
import numpy as np
import pandas as pd
from rapidfuzz import process, fuzz

from ZulipMessenger import reportError, reportStatus
from sentence_transformers import util
from resources.model import llm, timely_closing_ST_model
from resources.phrases import phrases_to_mark_met, survey_phrases, feedback_phrases, disconnect_phrases_en, \
    disconnect_phrases_hi, verbiage_phrases, hold_phrases, no_hold_phrases, duration_patterns, thank_you_phrases
from resources.prompts import (RudeSarcastic_prompt, escalation_prompt, Supervisor_prompt, prompt_closing, \
                               prompt_opening, Empathy_apology_prompt, reassurance_prompt,
                               Unethical_Solicitation_prompt,
                               voice_of_customer_prompt, prompt_opening_lang, timely_closing_prompt,
                               prompt_Personalization, DSAT_prompt)
from resources.result_extractor_cleaner import extract_json_objects, clean_text
from resources.working_with_files import validateDataframes

# Configure logging
ERROR_DUE_TO_LONG_CALL_TRANSCRIPT = "500"


def classify_rude_sarcastic(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):  # Faster than iterrows()
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{RudeSarcastic_prompt}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'Sarcasm_rude_behaviour': clean_text(extracted.get('Sarcasm_rude_behaviour', 'N/A')),
                'Sarcasm_rude_behaviour_evidence': clean_text(extracted.get('Sarcasm_rude_behaviour_evidence', 'N/A'))
            })
        except Exception as e:
            errors.append(request_id)
            results.append({'request_id': request_id, 'Sarcasm_rude_behaviour': "Error",
                            'Sarcasm_rude_behaviour_evidence': str(e)})

    return pd.DataFrame(results), errors


def process_transcripts_escalation(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{escalation_prompt}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'escalation_results': clean_text(extracted.get('Value', 'N/A')),
                'Issue_Identification': clean_text(extracted.get('Issue', 'N/A')),
                'Probable_Reason_for_Escalation': clean_text(extracted.get('Reason', 'N/A')),
                'Probable_Reason_for_Escalation_Evidence': clean_text(extracted.get('Evidence', 'N/A')),
                'Agent_Handling_Capability': clean_text(extracted.get('Agent Handling Capability', 'N/A')),
                'Escalation_Category': clean_text(extracted.get('Escalation Category', 'N/A'))
            })
        except Exception as e:
            errors.append(request_id)
            results.append({
                'request_id': request_id,
                'escalation_results': "Error",
                'Issue_Identification': str(e),
                'Probable_Reason_for_Escalation': str(e),
                'Probable_Reason_for_Escalation_Evidence': str(e),
                'Agent_Handling_Capability': str(e),
                'Escalation_Category': str(e)
            })

    return pd.DataFrame(results), errors


def classify_supervisor(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{Supervisor_prompt}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'Wanted_to_connect_with_supervisor': clean_text(extracted.get('Wanted_to_connect_with_supervisor',
                                                                              'N/A')),
                'de_escalate': clean_text(extracted.get('de_escalate', 'N/A')),
                'Supervisor_call_connected': clean_text(extracted.get('Supervisor_call_connected', 'N/A')),
                'call_back_arranged_from_supervisor': clean_text(extracted.get('call_back_arranged_from_supervisor',
                                                                               'N/A')),
                'supervisor_evidence': clean_text(extracted.get('supervisor_evidence', 'N/A')),
                'Denied_for_Supervisor_call': clean_text(extracted.get('Denied_for_Supervisor_call', 'N/A')),
                'denied_evidence': clean_text(extracted.get('denied_evidence', 'N/A'))
            })
        except Exception as e:
            errors.append(request_id)
            results.append({
                'request_id': request_id,
                'Wanted_to_connect_with_supervisor': "Error",
                'de_escalate': str(e),
                'Supervisor_call_connected': str(e),
                'call_back_arranged_from_supervisor': str(e),
                'supervisor_evidence': str(e),
                'Denied_for_Supervisor_call': str(e),
                'denied_evidence': str(e)
            })

    return pd.DataFrame(results), errors


def retry_classification(main_df, parameter_df, classify_func, error_ids, columns, max_retries=25):
    """
    Retries classification for failed request IDs up to a maximum of 20 times.

    Args:
        main_df (DataFrame): Original dataframe.
        parameter_df (DataFrame): DataFrame where results need to be updated.
        classify_func (function): Function used to classify the parameter.
        error_ids (list): List of request IDs that failed classification.
        columns (list): Expected columns in the classification result.
        max_retries (int): Maximum number of retry attempts (default: 20).

    Returns:
        DataFrame: Updated parameter_df with retried values.
    """
    attempt = 0
    initial_error_ids = set(error_ids)  # Store initial error IDs for final reporting

    while error_ids and attempt < max_retries:
        attempt += 1
        rerun_res_df, new_error_ids = classify_func(main_df, request_ids=error_ids)

        if not rerun_res_df.empty:
            # Update only failed request IDs in the existing dataframe
            for row in rerun_res_df.to_dict('records'):
                request_id = row['request_id']
                if request_id in parameter_df['request_id'].values:
                    row_series = pd.Series(row)  # Convert row to Pandas Series
                    parameter_df.loc[parameter_df['request_id'] == request_id, columns] = row_series[columns].values

        error_ids = new_error_ids  # Update remaining error IDs

        if not new_error_ids:
            return parameter_df  # Exit early if all errors are resolved

    # Report errors **only after 20 attempts are done**
    if error_ids:
        failed_ids = ", ".join(map(str, error_ids))
        reportError(f"❌ {classify_func} failed after {max_retries} attempts for {len(error_ids)} request IDs. "
                    f"Failed IDs: {failed_ids}")

    return parameter_df


def updating_RudeSarcasm_result(row, threshold=80):
    try:
        if str(row.get('Sarcasm_rude_behaviour', "")).strip() == "Not Met":
            evidence = row.get('Sarcasm_rude_behaviour_evidence', "").strip()

            if not evidence or not isinstance(phrases_to_mark_met, list):
                return row  # Skip processing if invalid input

            result = process.extractOne(evidence, phrases_to_mark_met, scorer=fuzz.partial_ratio)

            if result and isinstance(result, tuple) and len(result) > 1 and result[1] >= threshold:
                row['Sarcasm_rude_behaviour'] = "Met"

    except KeyError as e:
        reportError(f"❌ KeyError: Missing key {e} in request_id {row.get('request_id', 'Unknown')}")
    except AttributeError as e:
        reportError(f"⚠️ AttributeError: {e} in request_id {row.get('request_id', 'Unknown')}")
    except TypeError as e:
        reportError(f"⚠️ TypeError: {e} in request_id {row.get('request_id', 'Unknown')}")
    except ValueError as e:
        reportError(f"⚠️ ValueError: {e} in request_id {row.get('request_id', 'Unknown')}")
    except Exception as e:
        reportError(f"❗ Unexpected error: {e} in request_id {row.get('request_id', 'Unknown')}")

    return row


def detect_language(text):
    return langid.classify(text)[0] if isinstance(text, str) and text.strip() else "Unknown"


def aggregate_lang(agg_df):
    grouped = agg_df.groupby('request_id')

    def count_languages(series, speakers, target_speaker, target_language):
        return sum(
            1 for lang, speaker in zip(series, speakers) if lang == target_language and speaker == target_speaker)

    aggregated_data = []

    for header, group in grouped:

        # Get the filename for this group
        filename = group['conversation_id'].iloc[0]

        english_customer_count = count_languages(group['Detected_Language'], group['speaker'],
                                                 target_speaker="Customer",
                                                 target_language='English')
        hindi_customer_count = count_languages(group['Detected_Language'], group['speaker'], target_speaker="Customer",
                                               target_language='Hindi')
        english_agent_count = count_languages(group['Detected_Language'], group['speaker'], target_speaker="Agent",
                                              target_language='English')
        hindi_agent_count = count_languages(group['Detected_Language'], group['speaker'], target_speaker="Agent",
                                            target_language='Hindi')

        if hindi_customer_count > 0 and hindi_agent_count == 0:
            language_switch = "Customer spoke in Hindi but agent didn't switch language"
        elif hindi_customer_count > 0 and hindi_agent_count > 0:
            language_switch = "Agent switched language"
        else:
            language_switch = "Switched to English"

        aggregated_data.append({
            'conversation_id': filename,
            'request_id': header,
            'CustomerLangCount': f"English: {english_customer_count} || Hindi: {hindi_customer_count}",
            'AgentLangCount': f"English: {english_agent_count} || Hindi: {hindi_agent_count}",
            'language_switch': language_switch,
            'language_switch_category': language_switch
        })

    lang_df = pd.DataFrame(aggregated_data)

    return lang_df


def classify_langSwitch(transcriptChat_df):
    # Apply detection with progress bar
    transcriptChat_df['Detected_Language'] = transcriptChat_df['transcript'].apply(detect_language)

    # Standardize Speaker Labels
    transcriptChat_df['speaker'] = transcriptChat_df['speaker'].replace({'00': 'Agent', '01': 'Customer'})
    transcriptChat_df['Detected_Language'] = transcriptChat_df['Detected_Language'].apply(
        lambda x: 'English' if x == 'en' else 'Hindi')

    # Get the aggregated language switch dataframe
    lang_switch_df = aggregate_lang(transcriptChat_df)
    return lang_switch_df


def classifyApologyEmpathy(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):  # Faster than iterrows()
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{Empathy_apology_prompt}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'Apology_result': clean_text(extracted.get('Apology', 'N/A')),
                'Apology_evidence': clean_text(extracted.get('Apology Evidence', 'N/A')),
                'Empathy_result': clean_text(extracted.get('Empathy', 'N/A')),
                'Empathy_evidence': clean_text(extracted.get('Empathy Evidence', 'N/A')),
                'Apology_Category': clean_text(extracted.get('Apology Category', 'N/A')),
                'Empathy_Category': clean_text(extracted.get('Empathy Category', 'N/A'))
            })

        except Exception as e:
            errors.append(request_id)
            error_message = str(e)
            if ERROR_DUE_TO_LONG_CALL_TRANSCRIPT in error_message:
                results.append({
                    'request_id': request_id,
                    'Apology_result': "Error 500",
                    'Apology_evidence': "An unexpected error occurred on Google's side. Your input context is too long.",
                    'Empathy_result': "Error 500",
                    'Empathy_evidence': "An unexpected error occurred on Google's side. Your input context is too long.",
                    'Apology_Category': "Error 500",
                    'Empathy_Category': "An unexpected error occurred on Google's side. Your input context is too long."
                })
            else:
                print(f"Error processing request_id {request_id}: {e}")
                results.append({
                    'request_id': request_id,
                    'Apology_result': "Error",
                    'Apology_evidence': str(e),
                    'Empathy_result': "Error",
                    'Empathy_evidence': str(e),
                    'Apology_Category': "Error",
                    'Empathy_Category': str(e)
                })

    return pd.DataFrame(results), errors


def classifyUnethicalSolicitation(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):  # Faster than iterrows()
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{Unethical_Solicitation_prompt}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'Unethical_Solicitation': clean_text(extracted.get('Unethical_Solicitation', 'N/A')),
                'Unethical_Solicitation_Evidence': clean_text(extracted.get('Unethical_Solicitation_Evidence', 'N/A'))
            })

        except Exception as e:
            errors.append(request_id)
            error_message = str(e)
            if ERROR_DUE_TO_LONG_CALL_TRANSCRIPT in error_message:
                results.append({
                    'request_id': request_id,
                    'Unethical_Solicitation': "Error 500",
                    'Unethical_Solicitation_Evidence': "An unexpected error occurred on the server side.",
                })
            else:
                print(f"Error processing request_id {request_id}: {e}")
                results.append({
                    'request_id': request_id,
                    'Unethical_Solicitation': "Error",
                    'Unethical_Solicitation_Evidence': str(e)
                })

    return pd.DataFrame(results), errors


def classifyReassurance(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):  # Faster than iterrows()
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{reassurance_prompt}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'Reassurance_result': clean_text(extracted.get('Value', 'N/A')),
                'Reassurance_evidence': clean_text(extracted.get('Evidence', 'N/A')),
                'Reassurance_Category': clean_text(extracted.get('Category', 'N/A'))

            })

        except Exception as e:
            errors.append(request_id)
            error_message = str(e)
            if ERROR_DUE_TO_LONG_CALL_TRANSCRIPT in error_message:
                results.append({
                    'request_id': request_id,
                    'Reassurance_result': "Error 500",
                    'Reassurance_evidence': "An unexpected error occurred on the server side.",
                    'Reassurance_Category': str(e)
                })
            else:
                print(f"Error processing request_id {request_id}: {e}")
                results.append({
                    'request_id': request_id,
                    'Reassurance_result': "Error",
                    'Reassurance_evidence': str(e),
                    'Reassurance_Category': str(e)
                })

    return pd.DataFrame(results), errors


def classifyChatClosing(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):  # Faster than iterrows()
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{prompt_closing}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'Further Assistance': clean_text(extracted.get('Further Assistance', 'N/A')),
                'Further Assistance Evidence': clean_text(extracted.get('Further Assistance Evidence', 'N/A')),
                'Effective IVR Survey': clean_text(extracted.get('Effective IVR Survey', 'N/A')),
                'Effective IVR Survey Evidence': clean_text(extracted.get('Effective IVR Survey Evidence', 'N/A')),
                'Branding': clean_text(extracted.get('Branding', 'N/A')),
                'Branding Evidence': clean_text(extracted.get('Branding Evidence', 'N/A')),
                'Greeting': clean_text(extracted.get('Greeting', 'N/A')),
                'Greeting Evidence': clean_text(extracted.get('Greeting Evidence', 'N/A'))
            })

        except Exception as e:
            errors.append(request_id)
            error_message = str(e)
            if ERROR_DUE_TO_LONG_CALL_TRANSCRIPT in error_message:
                results.append({
                    'request_id': request_id,
                    'Further Assistance Evidence': "An unexpected error occurred on Google's side. Your input context "
                                                   "is too long.",
                    'Effective IVR Survey': "Error 500",
                    'Effective IVR Survey Evidence': "An unexpected error occurred on Google's side. Your input "
                                                     "context is too long.",
                    'Branding': "Error 500",
                    'Branding Evidence': "An unexpected error occurred on Google's side. Your input context is too "
                                         "long.",
                    'Greeting': "Error 500",
                    'Greeting Evidence': "An unexpected error occurred on Google's side. Your input context is too "
                                         "long."

                })
            else:
                print(f"Error processing request_id {request_id}: {e}")
                results.append({
                    'request_id': request_id,
                    'Further Assistance': "Error",
                    'Further Assistance Evidence': str(e),
                    'Effective IVR Survey': "Error",
                    'Effective IVR Survey Evidence': str(e),
                    'Branding': "Error",
                    'Branding Evidence': str(e),
                    'Greeting': "Error",
                    'Greeting Evidence': str(e)
                })

    return pd.DataFrame(results), errors


def classifyChatOpening(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):  # Faster than iterrows()
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{prompt_opening}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'Greeting_the_customer': clean_text(extracted.get('Greeting the Customer', 'N/A')),
                'Greeting_the_customer_evidence': clean_text(extracted.get('Greeting the Customer Evidence', 'N/A')),
                'Self_introduction': clean_text(extracted.get('Self Introduction', 'N/A')),
                'Self_introduction_evidence': clean_text(extracted.get('Self Introduction Evidence', 'N/A')),
                'Identity_confirmation': clean_text(extracted.get('Customer Identity Confirmation', 'N/A')),
                'Identity_confirmation_evidence': clean_text(
                    extracted.get('Customer Identity Confirmation Evidence', 'N/A'))
            })

        except Exception as e:
            errors.append(request_id)
            error_message = str(e)
            if ERROR_DUE_TO_LONG_CALL_TRANSCRIPT in error_message:
                results.append({
                    'request_id': request_id,
                    'Greeting_the_customer': "An unexpected error occurred on Google's side. Your input context "
                                                   "is too long.",
                    'Greeting_the_customer_evidence': "Error 500",
                    'Self_introduction': "An unexpected error occurred on Google's side. Your input "
                                                     "context is too long.",
                    'Self_introduction_evidence': "Error 500",
                    'Identity_confirmation': "An unexpected error occurred on Google's side. Your input context is too "
                                         "long.",
                    'Identity_confirmation_evidence': "Error 500"

                })
            else:
                print(f"Error processing request_id {request_id}: {e}")
                results.append({
                    'request_id': request_id,
                    'Greeting_the_customer': "Error",
                    'Greeting_the_customer_evidence': str(e),
                    'Self_introduction': "Error",
                    'Self_introduction_evidence': str(e),
                    'Identity_confirmation': "Error",
                    'Identity_confirmation_evidence': str(e)
                })

    return pd.DataFrame(results), errors


def classify_DSAT(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):  # Faster than iterrows()
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{DSAT_prompt}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'Customer_Issue_Identification': clean_text(extracted.get('Customer_Issue_Identification', 'N/A')),
                'Reason_for_DSAT': clean_text(extracted.get('Reason_for_DSAT', 'N/A')),
                'Suggestion_for_DSAT_Prevention': clean_text(extracted.get('Suggestion_for_DSAT_Prevention', 'N/A'))
            })

        except Exception as e:
            errors.append(request_id)
            error_message = str(e)
            if ERROR_DUE_TO_LONG_CALL_TRANSCRIPT in error_message:
                results.append({
                    'request_id': request_id,
                    'Customer_Issue_Identification': "Error 500",
                    'Reason_for_DSAT': "Error",
                    'Suggestion_for_DSAT_Prevention': str(e)})
            else:
                print(f"Error processing request_id {request_id}: {e}")
                results.append({
                    'request_id': request_id,
                    'Customer_Issue_Identification': "Error",
                    'Reason_for_DSAT': "Error",
                    'Suggestion_for_DSAT_Prevention': str(e)
                })

    return pd.DataFrame(results), errors


def create_final_DSAT_results(df, DSAT_res_df, Survey_IDS):
    final_DSAT_res_df = pd.DataFrame({'request_id': df['request_id']})
    final_DSAT_res_df['DSAT_result'] = final_DSAT_res_df['request_id'].apply(
        lambda x: 'Yes' if x in Survey_IDS.values else 'No'
    )
    final_DSAT_res_df = final_DSAT_res_df.merge(DSAT_res_df, on='request_id', how='left')
    final_DSAT_res_df['Reason_for_DSAT'] = final_DSAT_res_df['Reason_for_DSAT'].fillna('N/A')
    final_DSAT_res_df['Suggestion_for_DSAT_Prevention'] = final_DSAT_res_df['Suggestion_for_DSAT_Prevention'].fillna(
        'N/A')
    return final_DSAT_res_df


def classifyVoiceOfCustomer(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):  # Faster than iterrows()
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{voice_of_customer_prompt}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'VOC_Category': clean_text(extracted.get('Category', 'N/A')),
                'VOC_Core_Issue_Summary': clean_text(extracted.get('Core_Issue_Summary', 'N/A'))
            })

        except Exception as e:
            errors.append(request_id)
            error_message = str(e)
            if ERROR_DUE_TO_LONG_CALL_TRANSCRIPT in error_message:
                results.append({
                    'request_id': request_id,
                    'VOC_Category': "Error 500",
                    'VOC_Core_Issue_Summary': "An unexpected error occurred on Google's side. Your input context is too long."
                })
            else:
                print(f"Error processing request_id {request_id}: {e}")
                results.append({
                    'request_id': request_id,
                    'VOC_Category': "Error",
                    'VOC_Core_Issue_Summary': str(e)
                })

    return pd.DataFrame(results), errors


def classifyOpeningLang(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):  # Faster than iterrows()
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{prompt_opening_lang}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'Open the call in default language': clean_text(extracted.get('default_opening_lang', 'N/A')),
                'Open the call in default language evidence': clean_text(extracted.get('Evidence', 'N/A')),
                'Open the call in default language Reason': clean_text(extracted.get('Reason', 'N/A'))
            })

        except Exception as e:
            errors.append(request_id)
            error_message = str(e)
            if ERROR_DUE_TO_LONG_CALL_TRANSCRIPT in error_message:
                results.append({
                    'request_id': request_id,
                    'Open the call in default language': "Error 500",
                    'Open the call in default language evidence': str(error_message),
                    'Open the call in default language Reason': "Error"
                })
            else:
                print(f"Error processing request_id {request_id}: {e}")
                results.append({
                    'request_id': request_id,
                    'Open the call in default language': "Error",
                    'Open the call in default language evidence': str(error_message),
                    'Open the call in default language evidence Reason': "Error"
                })

    return pd.DataFrame(results), errors


def classifyTimelyClosing(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):  # Faster than iterrows()
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{timely_closing_prompt}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'transcript': transcript,
                'Category': clean_text(extracted.get('Category', 'N/A')),
                'Summary': clean_text(extracted.get('Summary', 'N/A')),
                'Supporting_Evidence': clean_text(extracted.get('Supporting_Evidence', 'N/A'))
            })

        except Exception as e:
            errors.append(request_id)
            error_message = str(e)
            if ERROR_DUE_TO_LONG_CALL_TRANSCRIPT in error_message:
                results.append({
                    'request_id': request_id,
                    'transcript': transcript,
                    'Category': "Error",
                    'Summary': "Server Error",
                    'Supporting_Evidence': "Error"
                })
            else:
                print(f"Error processing request_id {request_id}: {e}")
                results.append({
                    'request_id': request_id,
                    'transcript': transcript,
                    'Category': "Error",
                    'Summary': str(e),
                    'Supporting_Evidence': "Error"
                })

    return pd.DataFrame(results), errors


def evaluate_verbiage(time_diff, threshold):
    if isinstance(time_diff, (int, float)):
        return 'Met' if time_diff <= threshold else 'Not Met'
    else:
        return None  # If not int or float, mark as None


def evaluate_timely_closing(r):
    if all([r['Verbiage_1_result'] == 'Met', r['Verbiage_2_result'] == 'Met',
            r['Verbiage_3_result'] == 'Met']):
        return 'Met'
    else:
        return 'Not Met'


#Timely Closing Parameter
def processing_timely_closing(timely_closing_primary_info, timely_closing_transcript, timely_closing_transcript_chat,
                              timely_closing_survey_column_name):
    request_ids = timely_closing_transcript['request_id'].tolist()
    print("Checking for calls that are pitched...")
    call_ended_abruptly_ids = \
        timely_closing_primary_info[timely_closing_primary_info[timely_closing_survey_column_name].isnull()][
            'request_id'].tolist()
    if len(call_ended_abruptly_ids) == 0:
        print("All ids went for survey pitch and were closed timely.")
        # Create a DataFrame with the specified request IDs and fill the other columns with the required values
        timely_closing_res_df = pd.DataFrame({
            'request_id': request_ids,
            'timely_closing_result': ['Met'] * len(request_ids),
            'timely_closing_evidence': ['Timely closing guidelines followed'] * len(request_ids)
        })
        return timely_closing_res_df
    else:
        print("Processing calls that were not pitched...")
        timely_closing_transcript['request_id'] = timely_closing_transcript['request_id'].astype(str)
        call_ended_abruptly_ids = [str(call_ended_abruptly_id) for call_ended_abruptly_id in call_ended_abruptly_ids]
        timely_closing_transcript_new = timely_closing_transcript[
            timely_closing_transcript['request_id'].isin(call_ended_abruptly_ids)]
        # Encode the phrases
        survey_embeddings = timely_closing_ST_model.encode(survey_phrases, convert_to_tensor=True)
        feedback_embeddings = timely_closing_ST_model.encode(feedback_phrases, convert_to_tensor=True)

        # Function to find matching phrases based on cosine similarity
        def find_matching_phrases(text, phrase_embeddings, threshold=0.7):
            text_sentences = text.split(". ")
            text_embeddings = timely_closing_ST_model.encode(text_sentences, convert_to_tensor=True)
            matching_phrases = False
            for sentence_embedding in text_embeddings:
                # Compute cosine similarity between the sentence and all phrases
                similarities = util.pytorch_cos_sim(sentence_embedding, phrase_embeddings)
                # Get the maximum similarity score for this sentence
                max_similarity = similarities.max().item()
                if max_similarity > threshold:
                    matching_phrases = True
                    break
            return matching_phrases

        # Evaluate each transcript and update DataFrame
        def evaluate_transcripts(transcript_df, survey_phrase_embeddings, feedback_phrase_embeddings):
            results = []
            for trans_idx, r in transcript_df.iterrows():
                text = r['transcript']
                has_feedback = find_matching_phrases(text, feedback_phrase_embeddings)
                has_survey = find_matching_phrases(text, survey_phrase_embeddings)
                print("Survey or feedback phrase found for request Id: " + r['request_id'])
                results.append({
                    'request_id': r['request_id'],
                    'has_feedback_phrase': has_feedback,
                    'has_survey_phrase': has_survey
                })
            return pd.DataFrame(results)

        # Process the transcripts
        processed_transcripts = evaluate_transcripts(timely_closing_transcript_new, survey_embeddings,
                                                     feedback_embeddings)
        # Merge with the original transcript DataFrame
        timely_closing_transcript_new = timely_closing_transcript_new.merge(processed_transcripts, on='request_id',
                                                                            how='left')
        # Find IDs that do not have a survey phrase but have a feedback phrase
        # Remove IDs that also have survey phrases
        print("Filtering calls where feedback was asked but doesn't connected to IVR for feedback...")
        feedback_ids = timely_closing_transcript_new[timely_closing_transcript_new['has_feedback_phrase']][
            'request_id'].tolist()
        ids_without_survey_but_with_feedback = timely_closing_transcript_new[
            timely_closing_transcript_new['request_id'].isin(feedback_ids) &
            ~timely_closing_transcript_new['has_survey_phrase']
            ]['request_id'].tolist()
        # Filter the final transcript DataFrame to include only these IDs
        final_filtered_transcript = timely_closing_transcript_new[
            timely_closing_transcript_new['request_id'].isin(ids_without_survey_but_with_feedback)
        ]
        final_transcript_df = final_filtered_transcript[
            ['request_id', 'transcript', 'has_feedback_phrase', 'has_survey_phrase']].reset_index(drop=True)
        final_transcript_ids = final_transcript_df['request_id']

        print("Processing IDs to drop calls where customer agreed to give feedback or the feedback was incomplete...")
        timely_closing_res_df, timely_closing_error_ids = classifyTimelyClosing(final_transcript_df)
        # If there are errors, process only those IDs
        if timely_closing_error_ids:
            timely_closing_columns = ['Category', 'Summary', 'Supporting_Evidence']
            timely_closing_res_df = retry_classification(final_transcript_df, timely_closing_res_df,
                                                         classifyTimelyClosing,
                                                         timely_closing_error_ids, timely_closing_columns)

        # Define the phrases to check against
        reference_phrases = ['The customer agreed to give feedback', 'Incomplete feedback request', ' [N/A]']
        reference_embeddings = timely_closing_ST_model.encode(reference_phrases, convert_to_tensor=True)

        # Function to check if the category matches any reference phrase with a threshold of 0.9
        def matches_reference_category(category):
            category_embedding = timely_closing_ST_model.encode(category)
            cosine_scores = util.pytorch_cos_sim(category_embedding, reference_embeddings)
            return cosine_scores.max().item() > 0.9  # Using threshold of 0.9

        if timely_closing_res_df.empty:
            print("Data went Empty no category matched!!!")
            # Create a DataFrame with the specified request IDs and fill the other columns with the required values
            timely_closing_res_df = pd.DataFrame({
                'request_id': request_ids,
                'timely_closing_result': ['Met'] * len(request_ids),
                'timely_closing_evidence': ['Timely closing guidelines followed'] * len(request_ids)
            })
            return timely_closing_res_df
        else:
            # Filter based on similarity
            timely_closing_res_df = timely_closing_res_df[
                ~timely_closing_res_df['Category'].apply(matches_reference_category)
            ]
            # Display the filtered DataFrame
            if timely_closing_res_df.empty:
                print("Data went Empty as all calls were either went for survey pitch or ended abruptly")
                # Define the columns
                columns = ['request_id', 'timely_closing_result', 'timely_closing_evidence']
                # Create a DataFrame with the specified request IDs and fill the other columns with the required values
                timely_closing_res_df = pd.DataFrame({
                    'request_id': request_ids,
                    'timely_closing_result': ['Met'] * len(request_ids),
                    'timely_closing_evidence': ['Timely closing guidelines followed'] * len(request_ids)
                })
                return timely_closing_res_df
            else:
                print("There are some IDs where customer declined to give feedback!!!")
                timely_closing_transcript_chat = timely_closing_transcript_chat[
                    timely_closing_transcript_chat['request_id'].isin(final_transcript_ids)]

                # Define function to check for evidence phrase in transcript
                def check_phrases_in_transcript(trans_rows, evidence_start_times, evidence_transcript, threshold=0.8):
                    evidence_embedding = timely_closing_ST_model.encode(evidence_transcript, convert_to_tensor=True)
                    for r in range(len(trans_rows)):
                        # Encode each transcript row
                        transcript_embedding = timely_closing_ST_model.encode(trans_rows[r], convert_to_tensor=True)
                        # Compute cosine similarity
                        similarity = util.pytorch_cos_sim(evidence_embedding, transcript_embedding)
                        if similarity.item() >= threshold:
                            return {
                                'matched_string': trans_rows[r],
                                'starttime': evidence_start_times[r]
                            }
                    return None

                # Initialize new columns in DataModelling_res_df
                timely_closing_res_df['matched_string'] = None
                timely_closing_res_df['starttime'] = None
                # Iterate over each ID and check the evidence phrase
                for index, row in timely_closing_res_df.iterrows():
                    request_id = row['request_id']
                    evidence_phrase = row['Supporting_Evidence']
                    # Get the corresponding transcript data from transcript_chat
                    transcript_data = timely_closing_transcript_chat[
                        timely_closing_transcript_chat['request_id'] == request_id]
                    # Ensure 'transcript' and 'starttime' columns exist
                    if 'transcript' in transcript_data.columns and 'starttime' in transcript_data.columns:
                        transcript_rows = transcript_data['transcript'].tolist()
                        start_times = transcript_data['starttime'].tolist()
                        if transcript_rows:
                            # Check for evidence phrase in transcript rows
                            result = check_phrases_in_transcript(transcript_rows, start_times, evidence_phrase)
                            if result:
                                timely_closing_res_df.at[index, 'matched_string'] = result['matched_string']
                                timely_closing_res_df.at[index, 'starttime'] = result['starttime']
                                print(f"ID {request_id}: Evidence phrase found and recorded.")
                            else:
                                print(f"ID {request_id}: Evidence phrase not found in transcript.")
                        else:
                            print(f"ID {request_id}: Transcript rows are empty.")
                    else:
                        print(f"ID {request_id}: Missing required columns in transcript data.")
                # Encode disconnect phrases
                embedding_disconnect_en = [timely_closing_ST_model.encode(phrase, convert_to_tensor=True) for phrase in
                                           disconnect_phrases_en]
                embedding_disconnect_hi = [timely_closing_ST_model.encode(phrase, convert_to_tensor=True) for phrase in
                                           disconnect_phrases_hi]
                print("Fetching details when agent asked the customer to disconnect the call...")

                def check_disconnect_phrases(trans_rows, disconnect_time, embedding_disconnect_en_phrase,
                                             embedding_disconnect_hi_phrase, model,
                                             disconnect_start_time, threshold=0.5):
                    if disconnect_start_time is None:
                        print("Start time is None. Skipping check.")
                        return {'found': False, 'time': None}
                    for tr in range(len(trans_rows)):
                        combined_text = trans_rows[tr]
                        combined_start_time = disconnect_time[tr] if tr < len(disconnect_time) else None
                        if combined_start_time is not None and combined_start_time > disconnect_start_time:  # Ensure we check
                            # after the given start time
                            embedding_text = model.encode(combined_text, convert_to_tensor=True)
                            # Check against English phrases
                            for embedding_disconnect in embedding_disconnect_en_phrase:
                                similarity = util.pytorch_cos_sim(embedding_text, embedding_disconnect)
                                if similarity.item() >= threshold:
                                    return {'found': f'{combined_text}', 'time': combined_start_time}
                            # Check against Hindi phrases
                            for embedding_disconnect in embedding_disconnect_hi_phrase:
                                similarity = util.pytorch_cos_sim(embedding_text, embedding_disconnect)
                                if similarity.item() >= threshold:
                                    return {'found': f'{combined_text}', 'time': combined_start_time}
                    return {'found': None, 'time': None}

                # Initialize new columns for disconnect phrase detection
                timely_closing_res_df['disconnect_phrase_found'] = None
                timely_closing_res_df['disconnect_time'] = None
                # Iterate over each ID and check for disconnect phrases after the detected start time
                for index, row in timely_closing_res_df.iterrows():
                    request_id = row['request_id']
                    starttime = row['starttime']  # This is the start time after which we check for disconnect phrases
                    # Get the corresponding transcript data from transcript_chat
                    transcript_data = timely_closing_transcript_chat[
                        timely_closing_transcript_chat['request_id'] == request_id]
                    # Ensure 'transcript' and 'starttime' columns exist
                    if 'transcript' in transcript_data.columns and 'starttime' in transcript_data.columns:
                        transcript_rows = transcript_data['transcript'].tolist()
                        start_times = transcript_data['Endtime'].tolist()
                        if transcript_rows:
                            # Check for disconnect phrases after the given start time
                            result = check_disconnect_phrases(transcript_rows, start_times, embedding_disconnect_en,
                                                              embedding_disconnect_hi, timely_closing_ST_model,
                                                              starttime)
                            if result['found']:
                                timely_closing_res_df.at[index, 'disconnect_phrase_found'] = result['found']
                                timely_closing_res_df.at[index, 'disconnect_time'] = result['time']
                                print(f"ID {request_id}: Disconnect phrase found and recorded.")
                            else:
                                print(f"ID {request_id}: No disconnect phrase found after start time.")
                        else:
                            print(f"ID {request_id}: Transcript rows are empty.")
                    else:
                        print(f"ID {request_id}: Missing required columns in transcript data.")
                # Step 1: Rename the "Request_id" column to "requestid" in DataModelling_res_df
                timely_closing_primary_info.rename(columns={'Request_id': 'request_id'}, inplace=True)
                timely_closing_primary_info.rename(columns={'Time_duration_of_Call': 'Conversation_End_Time'},
                                                   inplace=True)
                # Step 2: Merge based on 'requestid'
                timely_closing_res_df = timely_closing_res_df.merge(
                    timely_closing_primary_info[['request_id', 'Conversation_End_Time']],
                    on='request_id', how='left')

                def calculate_time_difference(r):
                    disconnect_time = r['disconnect_time']
                    # Extract Conversation_End_Time from the Time_duration_of_Call column
                    conversation_end_time = r['Conversation_End_Time']
                    # Check if both times are float or int
                    if isinstance(disconnect_time, (float, int)) and isinstance(conversation_end_time, (float, int)):
                        return conversation_end_time - disconnect_time
                    else:
                        return None

                # Create a new column 'time_difference' in DataModelling_res_df
                print("Checking if call was disconnected after agent request to disconnect the call from customer "
                      "side within 5 seconds...")
                timely_closing_res_df['time_difference'] = timely_closing_res_df.apply(calculate_time_difference,
                                                                                       axis=1)
                # Filter the DataModelling_res_df for rows where the time difference is greater than 5 seconds
                timely_closing_res_df = timely_closing_res_df[
                    timely_closing_res_df['time_difference'].apply(
                        lambda x: x > 5 if isinstance(x, (float, int)) else False)]
                if timely_closing_res_df.empty:
                    print("Data went Empty!!!")
                    # Define the columns
                    columns = ['request_id', 'timely_closing_result', 'timely_closing_evidence']
                    # Create a DataFrame with the specified request IDs and fill the other columns with the required
                    # values
                    timely_closing_res_df = pd.DataFrame({
                        'request_id': request_ids,
                        'timely_closing_result': ['Met'] * len(request_ids),
                        'timely_closing_evidence': ['Timely closing guidelines followed'] * len(request_ids)
                    })
                    return timely_closing_res_df
                else:
                    # Define phrases for disconnection checks
                    # Initialize new columns in DataModelling_res_df using .loc
                    print("Checking if the agent used the closing verbiage as per the guidelines "
                          "for calls that were not disconnected on time")
                    for i in range(1, 4):
                        timely_closing_res_df.loc[:, f'disconnection_verbiage_{i}'] = pd.NA
                        timely_closing_res_df.loc[:, f'disconnection_verbiage_{i}_time'] = pd.NA

                    # Function to calculate semantic similarity using the transformer model

                    def is_similar(phrase, transcript_phrase, threshold=0.7):
                        # Encode both phrases
                        embeddings1 = timely_closing_ST_model.encode(phrase, convert_to_tensor=True)
                        embeddings2 = timely_closing_ST_model.encode(transcript_phrase, convert_to_tensor=True)
                        # Calculate cosine similarity
                        cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
                        # Return True if similarity score is greater than the threshold
                        return cosine_scores.item() > threshold

                    # Function to check disconnection phrases after the disconnect_time
                    def check_disconnection_phrases(disconnect_request_id, disconnect_time, disconnect_transcript_chat):
                        global i
                        filtered_transcript = disconnect_transcript_chat[disconnect_transcript_chat['request_id'] ==
                                                                         disconnect_request_id]
                        for dis_index, dis_row in timely_closing_res_df[timely_closing_res_df['request_id'] ==
                                                                        disconnect_request_id].iterrows():
                            time_offset = disconnect_time
                            found = False
                            # Check for each disconnection verbiage
                            for i in range(1, 4):
                                phrase_key = f'disconnection_verbiage_{i}'
                                next_phrase_time = time_offset + (5 if i == 1 else 3)
                                matching_rows = filtered_transcript[
                                    filtered_transcript['starttime'] >= next_phrase_time]
                                for _, match_row in matching_rows.iterrows():
                                    if is_similar(verbiage_phrases[phrase_key], match_row['transcript']):
                                        # Found a semantically matching phrase within the allowed time window
                                        timely_closing_res_df.loc[
                                            dis_index, phrase_key] = f'Found ({match_row["transcript"]})'
                                        timely_closing_res_df.loc[dis_index, f'{phrase_key}_time'] = match_row[
                                            'starttime']
                                        time_offset = match_row['starttime']  # Move the time forward for next phrase
                                        found = True
                                        break
                                if not found:
                                    # Phrase not found, mark as 'Not Found'
                                    timely_closing_res_df.loc[dis_index, phrase_key] = 'Not Found'
                                    timely_closing_res_df.loc[dis_index, f'{phrase_key}_time'] = 'None'
                                    break  # Stop checking further phrases if one is not found
                            # If conversation ends before completing phrases
                            if i < 3 and not found:
                                timely_closing_res_df.loc[
                                    dis_index, f'disconnection_verbiage_{i + 1}'] = ('Conversation ended before '
                                                                                     'disconnect verbiage')

                    # Apply the function to all rows in DataModelling_res_df
                    for idx, row in timely_closing_res_df.iterrows():
                        check_disconnection_phrases(row['request_id'], row['disconnect_time'],
                                                    timely_closing_transcript_chat)

                    # Check for valid numeric types (int, float) before calculating differences
                    def calculate_time_diff(row, start_col, end_col):
                        start_time = row[start_col]
                        end_time = row[end_col]
                        # If either start or end time is not a number, return None
                        if not (isinstance(start_time, (int, float)) and isinstance(end_time, (int, float))):
                            return None
                        return start_time - end_time

                    # Calculate 'time_diff_disconnect_to_verbiage_1'
                    timely_closing_res_df['time_diff_disconnect_to_verbiage_1'] = timely_closing_res_df.apply(
                        lambda row: calculate_time_diff(row, 'disconnection_verbiage_1_time', 'disconnect_time'), axis=1
                    )
                    # Calculate 'time_diff_verbiage_1_to_2'
                    timely_closing_res_df['time_diff_verbiage_1_to_2'] = timely_closing_res_df.apply(
                        lambda row: calculate_time_diff(row, 'disconnection_verbiage_2_time',
                                                        'disconnection_verbiage_1_time'), axis=1
                    )
                    # Calculate 'time_diff_verbiage_2_to_3'
                    timely_closing_res_df['time_diff_verbiage_2_to_3'] = timely_closing_res_df.apply(
                        lambda row: calculate_time_diff(row, 'disconnection_verbiage_3_time',
                                                        'disconnection_verbiage_2_time'), axis=1
                    )

                    # Adding multiple verbiage result columns with varying thresholds
                    timely_closing_res_df['Verbiage_1_result'] = timely_closing_res_df[
                        'time_diff_disconnect_to_verbiage_1'].apply(lambda x: evaluate_verbiage(x, 5))
                    timely_closing_res_df['Verbiage_2_result'] = timely_closing_res_df[
                        'time_diff_verbiage_1_to_2'].apply(lambda x: evaluate_verbiage(x, 3))
                    timely_closing_res_df['Verbiage_3_result'] = timely_closing_res_df[
                        'time_diff_verbiage_2_to_3'].apply(lambda x: evaluate_verbiage(x, 3))
                    # Adding the 'timely_closing_result' column

                    timely_closing_res_df['timely_closing_result'] = timely_closing_res_df.apply(
                        evaluate_timely_closing,
                        axis=1)

                    # Adding the 'timely_closing_evidence' column
                    # Adding the 'timely_closing_evidence' column
                    def generate_evidence(r):
                        if r['timely_closing_result'] == 'Met':
                            return "Timely closing guidelines followed"
                        evidence = []
                        # Check Verbiage 1
                        if r['Verbiage_1_result'] == 'Met':
                            evidence.append("Verbiage_1 followed")
                        elif r['time_diff_disconnect_to_verbiage_1'] is not None:
                            excess_1 = round(r['time_diff_disconnect_to_verbiage_1'] - 5, 3)  # Threshold for Verbiage_1
                            evidence.append(f"Verbiage_1 exceeded by {excess_1:.3f} seconds")
                        else:
                            evidence.append("Verbiage_1 time difference unavailable")
                        # Check Verbiage 2
                        if r['Verbiage_2_result'] == 'Met':
                            evidence.append("Verbiage_2 followed")
                        elif r['time_diff_verbiage_1_to_2'] is not None:
                            excess_2 = round(r['time_diff_verbiage_1_to_2'] - 3, 3)  # Threshold for Verbiage_2
                            evidence.append(f"Verbiage_2 exceeded by {excess_2:.3f} seconds")
                        else:
                            evidence.append("Verbiage_2 time difference unavailable")
                        # Check Verbiage 3
                        if r['Verbiage_3_result'] == 'Met':
                            evidence.append("Verbiage_3 followed")
                        elif r['time_diff_verbiage_2_to_3'] is not None:
                            excess_3 = round(r['time_diff_verbiage_2_to_3'] - 3, 3)  # Threshold for Verbiage_3
                            evidence.append(f"Verbiage_3 exceeded by {excess_3:.3f} seconds")
                        else:
                            evidence.append("Verbiage_3 time difference unavailable")
                        return "; ".join(evidence)

                    # Apply the updated function to generate the 'timely_closing_evidence' column
                    timely_closing_res_df['timely_closing_evidence'] = timely_closing_res_df.apply(generate_evidence,
                                                                                                   axis=1)
                    timely_closing_res_df = timely_closing_res_df[
                        ['request_id', 'timely_closing_result', 'timely_closing_evidence']]
                    timely_closing_res_df = timely_closing_transcript.merge(timely_closing_res_df, on='request_id',
                                                                            how='left')
                    # Fill NaN values with different values for each column
                    fill_values = {
                        'timely_closing_result': 'Met',  # Fill NaN in column A with 0
                        'timely_closing_evidence': 'Timely closing guidelines followed. '
                    }
                    timely_closing_res_df = timely_closing_res_df.fillna(fill_values)
                    return timely_closing_res_df[['request_id', 'timely_closing_result', 'timely_closing_evidence']]


def process_Hold_Parameter(df, context_lines=3):
    # Check if required columns exist
    required_columns = {'request_id', 'transcript', 'Holddiff'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_columns - set(df.columns)}")

    hold_data = []  # To store the rows where hold is found

    for row in df.itertuples(index=True, name="Row"):
        index = row.Index  # Extract the index
        transcript = str(row.transcript).lower()  # Ensure transcript is a string
        totalholdtime = row.totalholdtime

        # Check if any of the hold phrases are present and no_hold phrases are absent
        if any(phrase in transcript for phrase in hold_phrases) and not any(
                no_phrase in transcript for no_phrase in no_hold_phrases):
            # Get the surrounding context: few lines before the current row
            start_index = max(0, index - context_lines)
            context_transcript = " ".join(
                df.iloc[start_index:index + 1]['transcript'].dropna().astype(str).tolist())  # Safe conversion

            # Check if requested required duration pattern is found in hold_evidence
            hold_evidence = f"Agent requested the customer to be on hold with : {row.transcript}"
            matched_phrase = next((match.group() for pattern in duration_patterns if
                                   (match := re.search(pattern, hold_evidence, re.IGNORECASE))), None)

            if matched_phrase:
                requested_required_duration = True
                requested_required_duration_evidence = f"Agent mentioned hold time to be 1 minute with: {matched_phrase}"
            else:
                requested_required_duration = False
                requested_required_duration_evidence = ("Agent put the call on hold without mentioning 1-minute "
                                                        "duration.")

            # Check the next 5 rows for "thank you for holding" phrases
            end_index = min(len(df), index + 6)  # Ensure we don't go out of bounds
            next_transcripts = " ".join(
                df.iloc[index + 1:end_index]['transcript'].dropna().astype(str).tolist())  # Safe conversion
            thank_you_found = any(re.search(phrase, next_transcripts, re.IGNORECASE) for phrase in thank_you_phrases)
            thank_you_evidence = f"Agent properly closed the hold with: {next_transcripts}" if thank_you_found else "There was no evidence of Thank you Phrase. Agent did not closed the hold with proper guidelines."

            # Get holdiff values for the next 5 rows
            hold_durations_after_hold_request = pd.to_numeric(df.iloc[index + 1:end_index]['Holddiff'],
                                                              errors='coerce').tolist()
            if hold_durations_after_hold_request:
                hold_ended_in_required_duration = 'Met' if all(
                    val < 63 for val in hold_durations_after_hold_request if not pd.isna(val)
                ) else 'Not Met'
                exceeded_values = [val - 63 for val in hold_durations_after_hold_request if
                                   not pd.isna(val) and val >= 63]
                hold_ended_in_required_duration_evidence = (
                    f"There was hold more than one minute(duration exceeded by {exceeded_values} seconds.)"
                    if exceeded_values
                    else "Hold ended in required duration which is 1 minute. Agent made sure to finish the hold on time."
                )

            # Create a new row with the hold information
            hold_data.append({
                'request_id': row.request_id,
                'hold_reason': context_transcript,  # Capture lines before and the current line
                'hold_found': True,  # Hold request was found
                'hold_result_evidence': hold_evidence,  # Provide the transcript as evidence
                'requested_required_duration': requested_required_duration,  # True/False based on match
                'requested_required_duration_evidence': requested_required_duration_evidence,  # Matched phrase if found
                'hold_thank_you_found': thank_you_found,  # True/False
                'hold_thank_you_evidence': thank_you_evidence,  # Provide the relevant evidence if found
                'thankyou_context': next_transcripts,
                'hold_ended_in_required_duration': hold_ended_in_required_duration,  # True/False
                'hold_ended_in_required_duration_evidence': hold_ended_in_required_duration_evidence,  # Exceeded values
                'hold_durations_after_hold_request': hold_durations_after_hold_request,  # List of holdiff values
                "totalholdtime": totalholdtime
            })
        else:
            hold_data.append({
                'request_id': row.request_id,
                'hold_reason': "N/A",
                'hold_found': False,  # Hold request was found
                'hold_result_evidence': "N/A",  # Provide the transcript as evidence
                'requested_required_duration': False,  # True/False based on match
                'requested_required_duration_evidence': "N/A",  # Matched phrase if found
                'hold_thank_you_found': False,  # True/False
                'hold_thank_you_evidence': "N/A",  # Provide the relevant evidence if found
                'thankyou_context': "N/A",
                'hold_ended_in_required_duration': True,  # True/False
                'hold_ended_in_required_duration_evidence': "N/A",  # Exceeded values
                'hold_durations_after_hold_request': "N/A",  # List of holdiff values
                "totalholdtime": totalholdtime
            })

    # Return DataFrame with results, ensuring all columns exist even if no data is found
    return pd.DataFrame(hold_data, columns=[
        'request_id', 'hold_reason', 'hold_found', 'hold_result_evidence',
        'requested_required_duration', 'requested_required_duration_evidence',
        'hold_thank_you_found', 'hold_thank_you_evidence', 'thankyou_context',
        'hold_ended_in_required_duration', 'hold_ended_in_required_duration_evidence',
        'hold_durations_after_hold_request', "totalholdtime"
    ])


def aggregate_hold_data(hold_df):
    if hold_df.empty:
        return hold_df  # Return empty if no data

    # Aggregation dictionary
    agg_funcs = {
        'hold_reason': lambda x: ' | '.join(set(x.dropna().replace("N/A", np.nan).dropna())),
        'hold_found': 'any',  # True if any row has True
        'hold_result_evidence': lambda x: ' | '.join(set(x.dropna().replace("N/A", np.nan).dropna())),
        'requested_required_duration': 'any',
        'requested_required_duration_evidence': lambda x: ' | '.join(set(x.dropna().replace("N/A", np.nan).dropna())),
        'hold_thank_you_found': 'any',
        'hold_thank_you_evidence': lambda x: ' | '.join(set(x.dropna().replace("N/A", np.nan).dropna())),
        'thankyou_context': lambda x: ' | '.join(set(x.dropna().replace("N/A", np.nan).dropna())),
        'hold_ended_in_required_duration': lambda x: 'Met' if 'Not Met' not in x.dropna().replace("N/A",
                                                                                                  np.nan).dropna().tolist() else 'Not Met',
        'hold_ended_in_required_duration_evidence': lambda x: [
            val for sublist in x.dropna().replace("N/A", np.nan).dropna()
            for val in (sublist if isinstance(sublist, list) else [sublist])
        ],  # Flatten list
        'hold_durations_after_hold_request': lambda x: [
            val for sublist in x.dropna().replace("N/A", np.nan).dropna()
            for val in (sublist if isinstance(sublist, list) else [sublist])
        ],  # Flatten list
        "totalholdtime": 'first'
    }

    # Group by 'request_id' and apply aggregations
    aggregated_df = hold_df.groupby('request_id', as_index=False).agg(agg_funcs)

    return aggregated_df


def determine_hold_request_found(row):
    if row['totalholdtime'] == 0:
        return "Met"
    elif row['totalholdtime'] > 0 and not row['hold_found']:
        return "Not Met"
    elif row['hold_found']:
        if row['requested_required_duration'] and row['hold_thank_you_found']:
            return "Met"
        else:
            return "Not Met"
    return "NA"


def determine_hold_evidence(row):
    if row['totalholdtime'] == 0:
        return "Hold Was Not Required."
    elif row['totalholdtime'] > 0 and not row['hold_found']:
        return "Hold was required but no hold phrase found."
    elif row['hold_found']:
        return row['hold_result_evidence'] + " " + row['requested_required_duration_evidence'] + " " + row[
            'hold_thank_you_evidence']
    return "NA"


def categorize_hold(row):
    category_parts = []
    if row['totalholdtime'] == 0:
        return "Hold Was Not Required"
    else:
        # Append values based on different columns
        if row['hold_found'] == False:
            category_parts.append("Hold was required but Hold was Not Informed")
        if row['requested_required_duration'] == False:
            category_parts.append("Agent put the call on hold without mentioning 1-minute duration.")
        if row['hold_thank_you_found'] == False:
            category_parts.append("Thank You Phrase Missing")
        if row['hold_ended_in_required_duration'] == "Not Met":
            category_parts.append("Hold exceeded by required time")

    if category_parts:
        return ", ".join(category_parts)
    return "Hold Guidelines Followed"


def create_spacy_pipeline():
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Add the language detector to the pipeline
    nlp.add_pipe("language_detector", last=True)

    return nlp


def is_hindi_word(word):
    # Simple heuristic: check if the word is written in Devanagari script (Hindi script)
    return bool(re.match('[\u0900-\u097F]+', word))


def calculate_row_language_percentage_spacy(df):
    # Create spaCy pipeline with language detection
    nlp = create_spacy_pipeline()

    # Create lists to store the language data
    language_data = []

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        transcript = row['transcript']
        request_id = row['request_id']  # Get the request_id for the current row

        # Process the transcript with spaCy to detect language
        doc = nlp(str(transcript))

        # Tokenize the transcript and count Hindi/English words (excluding empty spaces)
        hindi_count = 0
        english_count = 0
        total_count = 0

        # Loop through each token in the document, ignoring empty spaces
        for token in doc:
            # Skip empty tokens (spaces and other non-word tokens)
            if token.is_space or token.is_punct:
                continue

            total_count += 1
            if is_hindi_word(token.text):
                hindi_count += 1
            elif token.is_alpha and not is_hindi_word(token.text):
                english_count += 1

        # Calculate the percentages
        hindi_percentage = (hindi_count / total_count) * 100 if total_count > 0 else 0
        english_percentage = (english_count / total_count) * 100 if total_count > 0 else 0
        if hindi_percentage > english_percentage:
            language = 'Hindi'
        else:
            language = 'English'

        # Append the result for each row with the request_id
        language_data.append([str(request_id), f"{language}"])

    # Create a DataFrame with language percentages for each row
    language_df = pd.DataFrame(language_data, columns=['request_id', 'Language'])

    return language_df


def classifyPersonalization(df: pd.DataFrame, request_ids=None):
    results, errors = [], []
    request_id_list = set(request_ids if request_ids else df["request_id"].tolist())

    for row in df.to_dict('records'):  # Faster than iterrows()
        request_id = row.get("request_id")
        if request_id not in request_id_list:
            continue

        try:
            transcript = row.get("transcript", "")
            response = llm.invoke(f"{transcript}\n\n\n\n\n\n{prompt_Personalization}")
            extracted = extract_json_objects(response.content)[0]

            results.append({
                'request_id': request_id,
                'Personalization_result': clean_text(extracted.get('Personalization_result', 'N/A')),
                'Personalization_Evidence': clean_text(extracted.get('Personalization_Evidence', 'N/A'))
            })

        except Exception as e:
            errors.append(request_id)
            error_message = str(e)
            if ERROR_DUE_TO_LONG_CALL_TRANSCRIPT in error_message:
                results.append({
                    'request_id': request_id,
                    'Personalization_result': "Error 500",
                    'Personalization_Evidence': str(e)})
            else:
                print(f"Error processing request_id {request_id}: {e}")
                results.append({
                    'request_id': request_id,
                    'Personalization_result': "Error",
                    'Personalization_Evidence': str(e)
                })

    return pd.DataFrame(results), errors


def process_TimelyOpening(dataframe):
    # Convert 'starttime' to numeric, forcing errors to NaN if conversion fails
    dataframe['starttime'] = pd.to_numeric(dataframe['starttime'], errors='coerce')

    # Extract the first occurrence of each unique request_id
    first_rows = dataframe.groupby("request_id").first().reset_index()

    # Ensure 'starttime' is numeric
    first_rows['starttime'] = pd.to_numeric(first_rows['starttime'], errors='coerce')

    # Handle NaN values in 'starttime' to avoid further errors
    first_rows['Delayed call opening'] = first_rows['starttime'].apply(
        lambda x: 'Met' if pd.notna(x) and x <= 3 else 'Not Met'
    )

    # Add the 'timely_opening_evidence' column
    first_rows['Delayed call opening evidence'] = first_rows['starttime'].apply(
        lambda x: f'Call opened within 3 seconds. Agent opened the call at {x}.'
        if pd.notna(x) and x <= 3
        else f'Call opened after 3 seconds. Agent opened the call at {x}.' if pd.notna(x) else 'Start time is missing.'
    )

    # Return only the necessary columns
    return first_rows[['request_id', 'Delayed call opening', 'Delayed call opening evidence']]


def process_classification(classification_func, df, expected_columns, classification_name):
    max_retries = 5
    retry_delay = 5
    """
    Handles classification with retries, validation, and error handling.
    """
    for attempt in range(1, max_retries + 1):
        print(f"Attempt {attempt}: Processing {classification_name}...")
        # reportStatus(f"Attempt {attempt}: Processing {classification_name}...")

        # Perform classification
        res_df, error_ids = classification_func(df)

        # Retry for failed request IDs
        if error_ids:
            print(f"⚠️ Retrying classification for failed request IDs...")
            reportStatus(f"⚠️ Retrying classification for failed request IDs...")
            res_df = retry_classification(df, res_df, classification_func, error_ids, expected_columns)

        # If classification completely fails, retry the entire DataFrame
        if res_df is None:
            print(f"⚠️ Attempt {attempt} failed: No valid output. Retrying entire DataFrame...")
            if attempt == max_retries:
                reportError(
                    f"❌ Max retries reached for {classification_name} classification. No valid output received.")
                return None
            time.sleep(retry_delay)
            continue

        # Validate Output
        is_valid, missing_cols, extra_cols = validateDataframes(res_df, expected_columns + ["request_id"])

        # Drop extra columns if any
        if extra_cols:
            print(f"⚠️ Dropping extra columns: {extra_cols}")
            res_df = res_df.drop(columns=extra_cols, errors="ignore")

        # If output is valid, break the retry loop
        if is_valid:
            print(f"✅ {classification_name} processing complete")
            reportStatus(f"✅ {classification_name} processing complete")
            return res_df

        print(f"⚠️ Attempt {attempt} failed: Missing columns [{missing_cols}] detected. Retrying entire DataFrame...")

        # If max retries are reached, log error and return None
        if attempt == max_retries:
            reportError(f"❌ Max retries reached for {classification_name} classification. Issues:\n"
                        f"- Missing Columns: {missing_cols}")
            return None

        time.sleep(retry_delay)


def process_hold_data(transcriptChat_df):
    hold_df = process_Hold_Parameter(transcriptChat_df)
    return aggregate_hold_data(hold_df)


def apply_hold_logic(final_hold_df):
    final_hold_df['hold_request_found'] = final_hold_df.apply(determine_hold_request_found, axis=1)
    final_hold_df['hold_evidence'] = final_hold_df.apply(determine_hold_evidence, axis=1)
    final_hold_df['hold_ended_in_required_duration'] = final_hold_df['hold_ended_in_required_duration'].astype(str)

    # Assign values when totalholdtime is 0
    final_hold_df.loc[final_hold_df['totalholdtime'] == 0, 'hold_ended_in_required_duration'] = 'NA'
    final_hold_df.loc[
        final_hold_df['totalholdtime'] == 0, 'hold_ended_in_required_duration_evidence'] = "Hold Was Not Required."
    final_hold_df.loc[final_hold_df['totalholdtime'] == 0, 'hold_durations_after_hold_request'] = "N/A"

    mask = final_hold_df['totalholdtime'] > 0
    final_hold_df.loc[mask, 'hold_ended_in_required_duration_evidence'] = final_hold_df.loc[mask].apply(
        lambda row: row['hold_ended_in_required_duration_evidence'] if row['hold_found'] else 'N/A', axis=1)
    final_hold_df.loc[mask, 'hold_durations_after_hold_request'] = final_hold_df.loc[mask].apply(
        lambda row: row['hold_durations_after_hold_request'] if row['hold_found'] else 'N/A', axis=1)
    final_hold_df.loc[mask, 'hold_ended_in_required_duration'] = final_hold_df.loc[mask].apply(
        lambda row: 'N/A' if row['hold_found'] is False else row['hold_ended_in_required_duration'], axis=1)

    final_hold_df['hold_ended_in_required_duration_evidence'] = final_hold_df[
        'hold_ended_in_required_duration_evidence'].apply(clean_text)
    return final_hold_df


def process_dead_air(primaryInfo_df, transcriptChat_df):
    dead_air_ids = primaryInfo_df[
        (primaryInfo_df['Total_instance_long_dead_Air'] > 0) | (primaryInfo_df['Total_instance_short_dead_Air'] > 0)
        ]['request_id']

    dead_air_data = []
    for request_id in dead_air_ids.unique():
        request_transcripts = transcriptChat_df[transcriptChat_df['request_id'] == request_id].copy()
        if request_transcripts.empty:
            continue

        request_transcripts['Prev_Endtime'] = request_transcripts['Endtime'].shift(1)

        for _, row in request_transcripts.iterrows():
            if row['Dear_Air_short'] == 1 or row['Dear_Air_long'] == 1:
                if pd.isna(row['Prev_Endtime']):
                    dead_air_timestamp = "N/A"
                    hold_diff = "N/A"
                    hold_requested_before_dead_air = "Met"
                else:
                    dead_air_start = row['Prev_Endtime']
                    dead_air_end = row['starttime']
                    dead_air_timestamp = f"[{dead_air_start} , {dead_air_end}]"
                    hold_diff = pd.to_numeric(row["Holddiff"], errors='coerce')
                    hold_requested_before_dead_air = "Not Met" if hold_diff > 10 else "Met"

                dead_air_data.append({
                    "request_id": request_id,
                    "Hold_requested_before_dead_air": hold_requested_before_dead_air,
                    "long_dead_air": hold_diff,
                    "dead_air_timestamp": dead_air_timestamp
                })

    return pd.DataFrame(dead_air_data)


def aggregate_dead_air_data(dead_air_df):
    if dead_air_df.empty:
        return dead_air_df

    def aggregate_hold_status(values):
        return "Not Met" if "Not Met" in values.values else "Met"

    return dead_air_df.groupby("request_id", as_index=False).agg({
        "Hold_requested_before_dead_air": aggregate_hold_status,
        "long_dead_air": lambda x: ' | '.join(set(x.replace("N/A", np.nan).dropna().astype(str))),
        "dead_air_timestamp": lambda x: ' | '.join(set(x.replace("N/A", np.nan).dropna().astype(str)))
    })


def merge_hold_and_dead_air(final_hold_df, dead_air_df):
    final_hold_df = final_hold_df.merge(dead_air_df, on="request_id", how="left")
    final_hold_df.fillna({
        "Hold_requested_before_dead_air": "Met",
        "long_dead_air": "N/A",
        "dead_air_timestamp": "N/A"
    }, inplace=True)

    final_hold_df.loc[final_hold_df['totalholdtime'] == 0, ['Hold_requested_before_dead_air', 'long_dead_air',
                                                            'dead_air_timestamp']] = ['Met', 'N/A', 'N/A']
    return final_hold_df


def categorize_hold_status(final_hold_df):
    final_hold_df['Hold_category'] = final_hold_df.apply(categorize_hold, axis=1)
    return final_hold_df
