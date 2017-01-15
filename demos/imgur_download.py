import requests
import shutil




def getImg(directory, link, idx):
    print ('Image link ',link)
    r = requests.get(link, stream=True)
    #file_name = link.split('/')[-1]
    file_name = str(idx)+".jpg"
    if (r.status_code == 200):
        with open(directory+'/'+file_name, 'wb') as out_file:
            shutil.copyfileobj(r.raw, out_file)


def test():
    link = '125.212.233.106:3000/images/02625cd6-9d67-4562-b7f1-f52fdc0e06d9/0-0.jpg'
    print link.split('/')[-1]



