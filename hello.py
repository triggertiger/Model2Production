#from utils.sql_data_queries import TrainDatesHandler
import os
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError

# Fetch credentials from environment variables
SCOPES = ['https://www.googleapis.com/auth/drive']
CLIENT_SECRETS = 'client_secret.json'
# creds = Credentials.from_authorized_user_file(CLIENT_SECRETS, 
#                                               scopes=SCOPES)
FILENAME = 'fraud_transactions.db'

# Authenticate using the service account
creds = None

if os.path.exists('token.json'):
    print('path found to token.json')
    creds = Credentials.from_authorized_user_file('token.json', SCOPES)
# If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            CLIENT_SECRETS, SCOPES
        )
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.json', "w") as token:
      token.write(creds.to_json())

try:
    service = build("drive", "v3", credentials=creds)
    FOLDER_ID = '1eF7_bF3_Ri3DR1OLnOzPCa7cFJzp_4Sf'  
    #FILE_ID = 

    # supported mimeTypes here: https://developers.google.com/drive/api/guides/mime-types 

    results = (
        service.files().list(fields=" files(id, name)").execute()
        )
    items = results.get("files", [])

    if not items:
        print('no files found')
        
    print("Files list: ") 
    for item in items:
        print(f'{item["name"]} ){item["id"]}')   

except HttpError as error:
    
    print(f"An error occurred: {error}")


# Your logic to access the Google Drive folder and fetch the data

# Example: Listing files in the folder
results = service.files().list(q=f"'{FOLDER_ID} in parents").execute()
items = results.get('files', [])

if not items:
    print('No files found.')
else:
    print('Files:')
    for item in items:
        print(f"{item['name']} ({item['id']})")

