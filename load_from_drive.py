from utils.sql_data_queries import TrainDatesHandler
import os
import io

from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload


# Fetch credentials from environment variables
SCOPES = ['https://www.googleapis.com/auth/drive']
CLIENT_SECRETS = 'client_secret.json'

FILENAME = os.getenv('DATABASE_FILE_NAME')

# Authenticate using the service account
creds = None

def gcp_auth_download():

    if os.path.exists('token.json'):
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
        FOLDER_ID = os.getenv('GCP_FOLDER_ID')  
        print(FOLDER_ID)
        print(type(FOLDER_ID))
    except HttpError as error:
        print(f"An error occurred: {error}")
    # access the Google Drive folder and fetch the data
    results = service.files().list(q=f"'{FOLDER_ID}' in parents").execute()
    items = results.get('files', [])

    if not items:
        print('No files found.')
        return
    
    print('Files:')
    for item in items:
        print(f"{item['name']} ({item['id']})")
    database_file = [d for d in items if d['name'].endswith('.db')]
    print(f'database file is: {database_file[0]["id"]}')
    db_file_id = database_file[0]['id']
    
    local_path = os.path.join('./tmp', os.getenv('DATABASE_FILE_NAME'))
    if not os.path.exists(local_path):

        try:
            request = service.files().get_media(fileId=db_file_id)
            with io.FileIO(local_path, 'wb') as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    print(f"Download {int(status.progress() * 100)}.")
            
            print(f'downloaded complete to: {local_path}')
        except HttpError as error:
            print(f"An error occurred: {error}")

if __name__ == "__main__":
    gcp_auth_download()
    user_data = TrainDatesHandler(date='2019-01-01')
    df = user_data.get_prediction_data()
    print(df.head())

