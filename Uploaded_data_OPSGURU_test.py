from datetime import datetime, timedelta

from fetchData2 import fetchSoftskillOpsguru, fetchBrcpOpsguru, fetchInteractionOpsguru, fetchRoster
from resources.working_with_files import createDfOpsguru


def getOpsguruResult():
    response = {}
    # Get yesterday's date in Indian format (DD-MM-YYYY)
    yesterday = datetime.now() - timedelta(days=1)
    yesterday_indian = yesterday.strftime('%d-%m-%Y')

    # Convert explicitly to Year-Month-Day (YYYY-MM-DD) format
    yesterday_ymd = yesterday.strftime('%Y-%m-%d')
    print(yesterday_ymd)  # Example Output: 2025-04-02

    softskill, softskillResponse = fetchSoftskillOpsguru(yesterday_ymd)
    response['softskill'] = softskillResponse
    # softskill =pd.read_csv(r'softskill_15April.csv')
    brcp, BrcpResponse = fetchBrcpOpsguru(yesterday_ymd)
    brcp = brcp.drop(['TL_Email_Id', 'Location'], axis=1)

    response['brcp'] = BrcpResponse
    # brcp=pd.read_csv(r'BRCP_15April.csv')
    interaction, interactionResponse = fetchInteractionOpsguru(yesterday_ymd)
    response['interaction'] = interactionResponse
    roster, rosterResponse = fetchRoster()
    response['roster'] = rosterResponse
    #softskill.to_excel("softskillTest.xlsx")
    #brcp.to_excel("brcoTest.xlsx")
    # interaction.to_excel("interactiontest.xlsx")
    # roster.to_excel("roster.xlsx")

    if all([df is not None and not df.empty for df in [softskill, brcp, interaction, roster]]):
        print("All DataFrames have data!")
        OpsGuru_df = createDfOpsguru(softskill, brcp, interaction, roster)
        OpsGuru_df.to_excel(f"Opsguru_data_{yesterday_indian}.xlsx")
    else:
        print("Some DataFrames are empty or None!")


getOpsguruResult()
