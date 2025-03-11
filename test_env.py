from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
import os
key = os.environ.get("FINANCIAL_API_KEY")
print("FINANCIAL_API_KEY:", key)
