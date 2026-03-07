import os
import sys
from dotenv import load_dotenv, find_dotenv
import panel as pn
pn.extension()

class Utils:
  def __init__(self):
    pass
  def get_api_key(self):
    _ = load_dotenv(find_dotenv())
    return os.getenv("UNSTRUCTURED_DATA_API_KEY")
    
  def get_url(self):
    _ = load_dotenv(find_dotenv())
    return os.getenv("UNSTRUCTURED_DATA_API_URL")
  
  def get_openai_api_key(self):
    _ = load_dotenv(find_dotenv())
    return os.getenv("OPENAI_API_KEY")

class upld_file():
    def __init__(self):
        self.widget_file_upload = pn.widgets.FileInput(accept='.pdf,.ppt,.png,.html,.epub', multiple=False)
        self.widget_file_upload.param.watch(self.save_filename, 'filename')
    
    def save_filename(self,_):
        if len(self.widget_file_upload.value) > 2e6:
            print("file too large. 2 M limit")
        else:
            self.widget_file_upload.save('./example_files/' + self.widget_file_upload.filename)
        #print(f"filename_ = {self.widget_file_upload.filename}")
        #print(f"length of value {len(self.widget_file_upload.value)}")
