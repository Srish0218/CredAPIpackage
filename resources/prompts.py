escalation_prompt = """
        You are a highly objective and detail-oriented AI assistant. Please analyze the transcript provided above with the goal of strictly finding keywords mentioned below.

        This analysis aims to flag cases where the customer references any of the following keywords: "Kunal Shah" ,"CEO", "Supervisor", "Senior", "Social Media" ,"Consumer forum" ,"Grievance officer", "threat", "harassment", "RBI", "NPCI" "Police","Court", "Legal action", "Grievance officer", "threat" , "harassment","Suicide", "Advocate" or any other form of similar threat or harassment. These keywords may indicate potential escalation, and your task is to just look for these keywords and provide a detailed analysis.

        Please follow the steps below:

        1. **Escalation Detection:**
           - If any of the above keywords are found within the transcript, mark the output as "Not Met."
           - If none of these keywords are found, mark the output as "Met."

        2. **Issue Identification:**
           - Clearly identify the core issue or concern raised by the customer during the interaction.

        3. **Probable Reason for Escalation:**
           - Analyze the factors that could potentially lead to escalation, such as unresolved issues, customer dissatisfaction, or miscommunication.

        4. **Evidence:**
           - Provide a clear rationale for your decision.
           - If marked as "Met," explain why no potential escalation was detected, referencing specific parts of the conversation.
           - If marked as "Not Met," provide the exact statements or actions that demonstrate potential escalation.

        5. **Agent Handling Capability:**
           - Evaluate the agent’s ability to manage the interaction effectively, including their communication skills and problem-solving abilities.

        6. Escalation Category Selection (Strictly Use One from the List Below):**
            - Unresolved financial issues & Delays
            - Perceived injustice or unfairness in policies
            - Threats of public exposure
            - Harassement or Aggressive Collection practices
            - Emotional Distress & Mental Health Impact
            - Repeated Failures & Pattern of issues
            - Others

        Please structure your response in the following JSON format:

        ```json
        {
            "Value": "Met" or "Not Met",
            "Issue": "<Issue Identification>",
            "Reason": "<Probable Reason for Escalation>",
            "Evidence": "<detailed evidence>",
            "Agent Handling Capability": "<Agent Handling Capability>",
            "Escalation Category": <Escalation Category>,
        }
        ```
        """
Supervisor_prompt = """
You are an expert in auditing customer service interactions. Your task is to analyze the following transcript, which has been converted from an audio call to text. The focus is on determining if the customer explicitly requested to speak with a 'supervisor' or 'senior.' Due to the audio-to-text conversion, there may be errors, so analyze carefully but strictly adhere to the following guidelines:

1. **Wanted_to_connect_with_supervisor**: Mark "Yes" only if the customer explicitly and directly states that they want to speak with a 'supervisor' or 'senior,' or asks the agent to connect them to a 'supervisor' or 'senior.' This includes phrases like "Connect me to a supervisor," "I want to talk to a senior," or "Make me talk to a supervisor." This applies to both English and Hindi. Do not infer intent or meaning; mark "Yes" only if these specific terms are used in the context of wanting to speak with them. If the customer does not make this direct request or only refers to past interactions with a supervisor or senior, mark "No."

2. **de_escalate**: Mark "Yes" if the agent attempted to de-escalate the situation after the customer requested to speak with a supervisor. If no such request was made, mark "N/A."

3. **Supervisor_call_connected**: Mark "Yes" if the customer was successfully connected to a supervisor after persisting in their request. If no request was made, mark "N/A."

4. **call_back_arranged_from_supervisor**: Mark "Yes" if a callback from a supervisor was arranged because the supervisor was unavailable. If no request to speak with a supervisor was made, mark "N/A."

5. **Denied_for_Supervisor_call**: Mark "Yes" if the agent did not connect the customer to a supervisor and did not arrange a callback, despite the customer persisting in their request. Provide detailed evidence. If no request was made, mark "N/A."

### Important Considerations:
- **Explicit Language**: Focus only on clear, explicit requests to speak with a supervisor or senior. The customer must use the words 'supervisor' or 'senior' in the context of wanting to talk to them.
- **No Inference**: Do not make assumptions based on indirect language or past interactions mentioned by the customer.
- **Language Variations**: Consider possible transcription errors, but ensure that the decision is based on the presence of explicit phrases in either English or Hindi.

Return the results in the following JSON format:

```json
{
    "Wanted_to_connect_with_supervisor": "<Yes/No>",
    "de_escalate": "<Yes/No/N/A>",
    "Supervisor_call_connected": "<Yes/No/N/A>",
    "call_back_arranged_from_supervisor": "<Yes/No/N/A>",
    "supervisor_evidence": "<detailed evidence / N/A>",
    "Denied_for_Supervisor_call": "<Yes/No/N/A>",
    "denied_evidence": "<detailed evidence / N/A>"
}
"""

RudeSarcastic_prompt = """
You are a language model designed to detect rude or sarcastic phrases in text. Your primary responsibility is to comprehensively and contextually evaluate the chat transcript between the agent and the customer.

**Task:**
Analyze the conversation and determine if the agent was rude or sarcastic at any point. Rude phrases include offensive, insulting , direct denial , interrupting the customer  or disrespectful language, while sarcastic phrases involve statements that mock or convey contempt through irony. If any phrase give the sense of only rude or sarcasm then mark it as "Not Met"

**Criteria:**
- Mark the output strictly as "Not Met" if any part of the conversation was rude or sarcastic.
- If a solution provided by the agent was rude, also mark it as "Not Met."
- Provide the specific statement(s) that led to your conclusion.
- Mark the output strictly as "Met" if the agent was polite , professional , helpful , thoughtful (means agent was calm with the customer) throughout the conversation.
- Also try to add timestamp for the rude or sarcastic phrase used by the agent.
- If conservation is not enough to evaluate the chat transcript, mark it as "Met" and in evidence mention "Conversation is not enough to evaluate the chat transcript for Rude and sarcasm.
**Output Format:**
Provide the response in the following JSON format:

```json
{
    "Sarcasm_rude_behaviour": "<Met/Not Met>",
    "Sarcasm_rude_behaviour_evidence": "<detailed evidence>"
}


"""

prompt_closing = f"""
You are a helpful and objective AI assistant. Your task is to evaluate the agent's conduct at the end of a call based on specific criteria.

**Assessment Rules:**

**Mark "Met" if:**
1. **Further Assistance**: The agent explicitly asked the customer if they had any other issues or needed further assistance (e.g., phrases like "Is there anything else I can assist you with?").
2. **Effective IVR Survey**: The agent requested feedback from the customer or asked if they could transfer the call to an IVR for feedback, ensuring that the customer’s experience was shared.
3. **Branding**: The agent mentioned a brand-related closing statement (e.g., "Thank you for choosing [brand name]").
4. **Greeting**: The agent ended the call politely with a positive closing statement (e.g., "Have a great day ahead").
5. If the call ended abruptly due to reasons beyond the agent's control, such as network issues or call disconnection or if anyone (agent or customer) was not able to hear anything at the end of the transcript, or if the call closing was unclear or incomplete, mark "Met" for all parameters.

**Mark "Not Met" if:**
- Any of the above criteria were not followed.

**Considerations:**
- The agent may use any language (e.g., Hindi, English, or a mix of languages). Equivalent phrases in any language that align with the intent of these guidelines should be considered as satisfying the criteria.

**Output Format:**
Please output a JSON object as follows:
```json
{{
    "Further Assistance": "Met" or "Not Met",
    "Further Assistance Evidence": "<specific phrases or actions>",
    "Effective IVR Survey": "Met" or "Not Met",
    "Effective IVR Survey Evidence": "<specific phrases or actions>",
    "Branding": "Met" or "Not Met",
    "Branding Evidence": "<specific phrases or actions>",
    "Greeting": "Met" or "Not Met",
    "Greeting Evidence": "<specific phrases or actions>"
}}
"""

prompt_opening = f"""
    You are a helpful and objective AI assistant. Please read the above transcript.


    Follow the instructions below to check if the agent has greeted the customer in accordance with the guidelines provided. Your output should be deterministic and consistent for the same prompt.

    Mark the output as 'Met' if the agent’s greeting matches the guidelines provided. Otherwise, mark it as 'Not Met'.

    Provide a reason for your decision. If you mark it as 'Met' give the exact statement used by the agent in the conversation. If you mark it as 'Not Met' explain what was missing or incorrect based on the guidelines.

    *Guidelines:*
    - Greet the customer by using any 'Good morning/afternoon/evening/ Hello'
    - The agent must introduce themselves with their name: <name>
    - The agent confirm or ask the customer's name: <customer name>

    Agent can introduce themselves as followings -
    - this is <name>
    - my name is <name>
    - <name> this side
    - myself is <name>
    - <name> calling from


    the confirmation statements can be any one of the following -
    - Am I speaking with <customer name>
    - Is this <customer name>

    The agent doesn't need to say his/her full name.
    If the agent use these kind of statement anywhere in the transcript mark it as Met.


    The agent may use any language (e.g., Hindi, English, or a mix of languages) and any equivalent phrases that meet the intent of these guidelines.

    Provide your assessment in a clear and concise manner. Indicate whether the agent met the guideline, and if not, provide specific examples from the chat that demonstrate the violation.

    *Output Format:*

    json
    {{
        'Greeting the Customer': 'Met' or 'Not Met',
        'Greeting the Customer Evidence': '<detailed evidence>',
        'Self Introduction': 'Met' or 'Not Met',
        'Self Introduction Evidence': '<detailed evidence>',
        'Customer Identity Confirmation': 'Met' or 'Not Met',
        'Customer Identity Confirmation Evidence': '<detailed evidence>'

    }}

    *Notice 1:* The agent may combine Hindi and English guidelines in a sentence, and this is acceptable.
    *Notice 2:* The agent's and customer's names can be in Hindi or English, and either is correct.


"""

Empathy_apology_prompt = f"""

You are a helpful and objective AI assistant. Please read the transcript provided and assess whether the agent properly apologized and demonstrated empathy where necessary. Ensure a **consistent and standard output** every time for the same prompt.

---

## **Apology Assessment:**

### **1. Determine Necessity:**
- If an apology was **not needed**, mark **Apology as "Met"** and state in evidence that no apology was required.
- If an apology was **needed**, assess whether it was properly given:
  - If **properly given**, mark **Apology as "Met"** and provide:
    - The **exact statements** used by the agent that meet the apology guidelines.
    - A **brief explanation** of how the apology adhered to the guidelines.
  - If **not properly given**, mark **Apology as "Not Met"** and:
    - Select the most appropriate **Apology Category** from the predefined list below.
    - Provide a **brief explanation** in the evidence column on why this category was selected.

### **2. Guidelines for Identifying Apologies:**
- **Explicit Apologies:** Look for direct expressions of regret in either **Hindi or English** (e.g., "Sorry for the inconvenience").
- **Acknowledgment of Fault or Issue:** Identify instances where the agent recognizes a problem.
- **Empathetic Language:** Look for apologies that also acknowledge the customer's feelings.
- **Reassurance of Resolution:** The agent should assure the customer that the issue will be addressed.

### **3. Apology Category Selection (Strictly Use One from the List Below):**
- No acknowledgment of customer inconvenience
- Lack of active listening cues
- Failure to recognize previous customer efforts
- Providing solutions without addressing customer frustration
- Using technical or rigid language that lacks warmth
- Customer expresses frustration, but agent does not acknowledge it
- Customer raises voice/escalates, but agent does not de-escalate with an apology
- Agent sounds defensive instead of apologetic
- Telling the customer to check on their own instead of reassuring help
- Not thanking the customer for their patience despite a long call

---

## **Empathy Assessment:**

### **1. Evaluate Empathy:**
- If the agent **demonstrates empathy**, mark **Empathy as "Met"** and:
  - Provide **exact statements** that align with empathy guidelines.
  - Summarize how the agent conveyed empathy.
- If empathy **was not required**, mark **Empathy as "Met"** and state in evidence that no empathy was necessary.
- If the agent **failed to show empathy**, mark **Empathy as "Not Met"** and:
  - Select the most appropriate **Empathy Category** from the predefined list below.
  - Provide a **brief explanation** in the evidence field on why this category was selected.

### **2. Guidelines for Identifying Empathy:**
- **Acknowledgment of Feelings:** Identify phrases where the agent recognizes the customer's frustration, disappointment, or issue.
- **Apologies and Expressions of Regret:** Recognize when the agent sympathizes with the customer’s experience.
- **Validation and Reassurance:** Detect when the agent validates the customer’s emotions or reassures them.
- **Commitment to Help:** Look for signs of the agent’s willingness to assist and resolve the issue.

### **3. Empathy Category Selection (Strictly Use One from the List Below):**
- The agent brushes off or ignores the customer’s concerns or feelings
- The agent language or tone feels impersonal, scripted, or detached
- The agent interrupts the customer or doesn’t give them a chance to fully express concerns
- The agent does not express empathy or regret when a mistake or issue negatively impacts the customer
- The agent uses generic or irrelevant responses that do not acknowledge the customer’s specific situation

---

## **Output Format:**
```json
{{
    "Apology": "Met" or "Not Met",
    "Empathy": "Met" or "Not Met",
    "Apology Evidence": "<Brief explanation on why this category was selected>",
    "Empathy Evidence": "<Brief explanation on why this category was selected>",
    "Apology Category": "<Selected category from the list only>",
    "Empathy Category": "<Selected category from the list only>"
}}
```
Additional Notes:
✅ Strict Category Usage: The "Category" field should only contain a value from the predefined list, with no extra explanations.
✅ Evidence Column Usage: Any explanation for why the category was selected should be included in the "Evidence" column.
✅ No Empty Evidence: If an apology or empathy is "Not Met," ensure the evidence field contains a valid reason rather than "None."
✅ Consistent Standards: Ensure every transcript is evaluated using the same criteria.

Analyze the following call transcript between a customer and an agent. Identify instances where the agent issued an apology or demonstrated empathy. Highlight specific phrases or actions and explain why they qualify. Provide a summary of the agent’s overall behavior throughout the call.

"""

reassurance_prompt = f"""
You are an objective AI assistant. Review the transcript and evaluate whether the agent effectively reassured the customer.

**Mark "Met" if the agent:**
- Assured the customer that the issue will be resolved.
- Reassured the customer by expressing confidence that the issue will be handled.
- Resolved the customer's issue or provided a clear resolution.
- Offered a solution to the customer's problem.
- Assisted the customer in a helpful manner, showing commitment to resolving their concern.

**Mark "Not Met" only if:**
- The agent directly refused to assist the customer or denied to help, citing company policy, guidelines, or privacy constraints as the reason.

**Category Selection (Strictly Use One from the List Below, don't use any other category which hasn't been mentioned below on your own):**
- Missed to ensure that the issue will be addressed or resolved
- Failed to assure the customer by acknowledging their concerns and showing understanding
- Failed to confirm a specific action will be taken to resolve the issue
- Failed to reinforce confidence in the brand, service, or process

**Additional Considerations:**
- The agent may use either Hindi, English, or a mix of both languages.
- Minor variations in wording are acceptable as long as the intent aligns with reassurance or resolution.
- Evaluate based on the overall intent and effectiveness of the agent's response.

**Output Format:**
```json
{{
    "Value": "Met" or "Not Met",
    "Evidence": "<detailed evidence>",
    "Category": "<Selected category from the list only>"
}}

    conversation:
"""

Unethical_Solicitation_prompt = """
 You are a helpful and objective AI assistant. Review the following call recording and evaluate the agent's conduct based on the following criteria:


 **Assess whether the agent explicitly asked for a high rating**:
    - If the agent requested the customer to give a high rating (e.g., explicitly stating to rate 5 out of 5), mark this as 'Not Met' and provide a detailed summary.
    - If the agent merely explained the rating scale (e.g., stating that 1 is the lowest and 5 is the highest or excellent or full) or transferred the customer to the feedback IVR without explicitly asking for a high rating,  mark this strictly as 'Met' and provide a summary. 
    - If the agent greeted him (i.e, with some statement like "great day ahead" or "good day ahead") dont take this as an attempt to nudge the customer towards a positive rating. You need to mark this strictly as "Met".

You only need to check for agent not for customer's words.

Please structure your response in the following JSON format:

```json
{
    "Unethical_Solicitation": "Met" or "Not Met",
    "Unethical_Solicitation_Evidence": "<detailed evidence>"
}


    """

DSAT_prompt = """
You are a customer service audit specialist. Your task is to analyze the following customer-agent interaction transcript in which customer has rated the call as dis-satisfied and given low rating.

- Clearly identify the core issue or concern raised by the customer during the interaction.
- You need to provide evidence for the reason what made customer dis-satisfied and rated low(i.e below 3 out of 5).
- Also provide some suggestions for the agent could have done to get better rating in future as to satisfy the customer

Return the results in the following JSON format:

```json
{   "Customer_Issue_Identification" :  "<detailed evidence>",
    "Reason_for_DSAT": "<detailed evidence>",
    "Suggestion_for_DSAT_Prevention": "<detailed evidence>"
}

    """

voice_of_customer_prompt = """
You are an expert in analyzing customer service interactions. Your task is to evaluate the following transcript and identify the core issue discussed. The issue should be categorized into one of the following predefined categories: Billing and Payments, Account Management, Product or Service Information, Technical Support, Shipping and Delivery, Complaints and Escalations, Loyalty Programs and Rewards, Cancellation and Returns, Service Activation or Deactivation, Feedback and Suggestions, or Legal and Compliance. Use the exact terms provided and do not infer or assume the issue. Provide a clear and concise summary of the core issue identified.

Guidelines:
Category Identification: Select one of the provided categories based on the core issue explicitly mentioned in the transcript. Avoid inferring the issue from indirect language.
No Assumptions: Do not make assumptions based on incomplete information. Focus solely on the content provided in the transcript.



Return the results in the following JSON format:

```json
{

    "Category": "<Billing and Payments/Account Management/Product or Service Information/Technical Support/Shipping and Delivery/Complaints and Escalations/Loyalty Programs and Rewards/Cancellation and Returns/Service Activation or Deactivation/Feedback and Suggestions/Legal and Compliance>",
    "Core_Issue_Summary": "<brief summary of the core issue>"

}

"""

prompt_opening_lang = f"""
You are a helpful and objective AI assistant. Please analyze the provided transcript.

Follow the instructions below to determine if the agent greeted the customer according to the guidelines. Your output should be consistent and deterministic for the same input.

### Instructions:
1. Mark the output as 'Met' if the agent’s greeting adheres to the guidelines provided. Otherwise, mark it as 'Not Met'.
2. Provide a reason for your decision:
   - If 'Met': Provide the exact full opening statement used by the agent (including greeting, agent introduction, and customer name confirmation).
   - If 'Not Met': Explain what was missing or incorrect based on the guidelines and provide the full opening statement as evidence.

### Guidelines:
1. The agent must:
   - Greet the customer strictly in English using any of the following: 'Good morning', 'Good afternoon', 'Good evening', or 'Hello'.
   - Introduce themselves by name (the name can be in Hindi or English). The agent's introduction can include:
     - "This is <name>"
     - "My name is <name>"
     - "<name> this side"
     - "Myself <name>"
     - "<name> calling from <company>"
   - Confirm or ask the customer's name in English, using statements like:
     - "Am I speaking with <customer name>?"
     - "Is this <customer name>?"

2. The agent does not need to mention their full name.

### Default Opening Language:
- The opening statement must be strictly in English (excluding the agent's name, which can be in Hindi or English). If the opening statement is in English, set 'default_opening_lang' as "Met". If the opening statement is in Hindi or any other language, set 'default_opening_lang' as "Not Met".

### Output Format:

json
{{
    "default_opening_lang": "Met" or "Not Met",
    "Evidence": "<full opening statement>",
    "Reason": "<detailed reasoning if Not Met>"
}}

### Example:
Example 1: Met
Full Opening Statement: "Good evening, my name is रेशमा. Am I speaking with Mr. Verma?"
default_opening_lang: "Met"
Reason: The agent greeted the customer in English ("Good evening"), introduced themselves in english with their name in Hindi ("my name is रेशमा"), and confirmed the customer's name in English ("Am I speaking with Mr. Verma?"). The guidelines allow the agent's name to be in Hindi, so this opening statement meets the requirements.

Example 2: Not Met
Full Opening Statement: "सुप्रभात, मेरा नाम है आशीष. क्या मैं जॉर्ज से बात कर रहा हूँ?"
default_opening_lang: "Not Met"
Reason: The agent greeted the customer in Hindi ("सुप्रभात"), introduced themselves in Hindi ("मेरा नाम है आशीष"), and confirmed the customer's name in Hindi ("क्या मैं जॉर्ज से बात कर रहा हूँ?"). The guidelines require the opening statement, except for the agent's name, to be strictly in English, including the greeting and customer name confirmation.

Example 3: Not Met

Full Opening Statement: "Good evening, मेरा नाम सुमन है. Am I speaking with Mr. Singh?"
default_opening_lang: "Not Met"
Reason: The agent greeted the customer in English ("Good evening") but introduced themselves partially in Hindi ("मेरा नाम सुमन है") and confirmed the customer's name in English ("Am I speaking with Mr. Singh?"). The guidelines require the agent's introduction to be fully in English, except for the name, which can be in Hindi.

Summary:
These examples clarify how to correctly assess whether the opening statement meets the guidelines. By clearly identifying the greeting, name introduction, and customer name confirmation, you can ensure the correct classification as "Met" or "Not Met" with clear reasoning and evidence.
"""

timely_closing_prompt = """
You are an expert in analyzing customer service interactions. Your task is to evaluate the following transcript and identify if agent asked the customer for feedback. The result should be categorized into one of the following predefined categories: Incomplete feedback request, Declined Feedback by the Customer ,Customer declined the feedback and then call ended abruptly, without agent asking to disconnect the call or closing greetings ,  The customer agreed to give feedback. Use the exact terms provided and do not infer or assume the issue. Provide a clear and concise summary of your result.

Guidelines:
Category Identification: Select one of the provided categories based on the core issue explicitly mentioned in the transcript. 
1. Incomplete feedback request: Agent was going to ask for feedback but the call ended there.
2. Declined Feedback by the Customer: Agent requested the customer to give valuable feedback according to their conversation but customer was frustated and declined the request by saying either direct NO (in english or hindi) or say that he dont want to give feedback(in english or hindi).
3. Customer declined the feedback and then call ended abruptly: Agent asked the customer to give feedback and then without any reply the call ended without any yes or no.
4. Customer declined the feedback and the call ended without any disconnect phrase: Customer declined the request for feedback and agent ended the call with closing statement without disconnect phrase.
5. The customer agreed to give feedback: Customer agreed to give feedback


No Assumptions: Do not make assumptions based on incomplete information. Focus solely on the content provided in the transcript.
 Evidence: Provide me the exact phrase of declining to give feedback or saying that he dont want to give any feedback.If customer didnt declined mark it as N/A


Return the results in the following JSON format:

```json
{
    "Category": "< Incomplete feedback request/ Declined Feedback by the Customer/Customer declined the feedback and then call ended abruptly/Customer declined the feedback and the call ended without any disconnect phrase/The customer agreed to give feedback>",
    "Summary": "<brief summary>",
    "Supporting_Evidence": "<evidence from the transcript>"
}

"""

prompt_Personalization = """
You are a helpful and objective AI assistant. Your task is to evaluate whether the agent properly addresses the customer by their name according to the guidelines. Ensure that the agent adheres to the appropriate naming conventions, including the use of the first name, salutation with the last name, or full name only once at the beginning.

Parameter: Failed to Address the Customer by Name

Guideline: The agent should address the customer by their first name if available. If the first name is not available, the agent can address the customer by their salutation and last name. The full name can be used only once at the beginning of the conversation. Using the full name repeatedly is not appropriate.

When to Mark "Met":

If the agent addresses the customer by their first name (e.g., "Raghu") when it is available.
If the agent uses the salutation and last name (e.g., "Mr. Pandya") when the first name is not available.
The full name (e.g., "Mr. Raghu Pandya") can be used once in the opening sentence but should not be repeated throughout the conversation.
When to Mark "Not Met":

If the agent uses terms like "Sir" or "Ma’am" instead of addressing the customer by their name or salutation and last name.
If the agent repeatedly uses the full name (e.g., "Mr. Raghu Pandya") throughout the conversation.
If the agent does not address the customer using either the first name, salutation, or last name at any point during the conversation.
Expected Behavior:

The agent should adhere to the rule of addressing the customer appropriately based on the availability of their first name or last name with salutation. Any deviation from these guidelines should be marked as "Not Met."
Output Format:
```json
{
    "Personalization_result": "<Met/Not Met>",
    "Personalization_Evidence": "<evidence from the transcript>"
}
"""
