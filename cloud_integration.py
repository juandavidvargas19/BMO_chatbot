from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import json
import os

class DriveUploader:
    def __init__(self, credentials_path, folder_id=None):
        self.credentials = Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/drive.file']
        )
        self.service = build('drive', 'v3', credentials=self.credentials)
        self.folder_id = folder_id
    
    def upload_jsonl(self, local_file_path, drive_filename=None):
        if not drive_filename:
            drive_filename = os.path.basename(local_file_path)
        
        # Check if file exists in Drive
        existing_file = self.find_file(drive_filename)
        
        media = MediaFileUpload(local_file_path, mimetype='application/json')
        
        if existing_file:
            # Update existing file
            file = self.service.files().update(
                fileId=existing_file['id'],
                media_body=media
            ).execute()
        else:
            # Create new file
            file_metadata = {'name': drive_filename}
            if self.folder_id:
                file_metadata['parents'] = [self.folder_id]
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
        
        return file.get('id')
    
    def find_file(self, filename):
        query = f"name='{filename}'"
        if self.folder_id:
            query += f" and parents in '{self.folder_id}'"
        
        results = self.service.files().list(q=query).execute()
        items = results.get('files', [])
        return items[0] if items else None