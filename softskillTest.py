from main import get_softskill_result

print(get_softskill_result())
# from datetime import datetime, timedelta
#
# import pandas as pd
# import pytz
#
# from fetchData import upload_softskill_result_on_database
#
# ist = pytz.timezone('Asia/Kolkata')
# date = (datetime.now(ist) - timedelta(days=1)).date()
# print("req date in IST:", date)
# df = pd.read_excel("ccredfinalOutput.xlsx")
# df["uploaded_date"] = date
#
# response = upload_softskill_result_on_database(df, date)
# print(response)
