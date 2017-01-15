import requests
import shutil




def getImg(directory, link, idx):
    r = requests.get(link, stream=True)
    if (r.status_code == 200):
        with open(directory+'/'+str(idx) +'.jpg', 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)
