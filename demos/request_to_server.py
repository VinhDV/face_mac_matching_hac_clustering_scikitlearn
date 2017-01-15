import requests
import json
url_reqbin = 'http://requestb.in/16vo0v91'
id = "11483427936188"

def send_cv_result(frameid, macs):
    '''
    Send map image-mac result to server.
    :param frameid: the id of each frame
    :param macs: array of macs address  (example: ['AB-CD', 'EF-GH'])
    with macs[0] to 0.png
    :return: response of http request
    <response200>
    '''
    url = 'http://125.212.233.106:3000/api/result'
    payload = {
      "id": frameid,
      "macs": macs
    }
    headers = {"Content-Type": "application/json"}
    payloadjs = json.dumps(payload)
    response = requests.post(url, data=payloadjs, headers=headers)
    return response

def test_cv_result():
    frameid = '11483427936188'
    macs = ['AB-AC', 'AD-AF']
    res = send_cv_result(frameid, macs)
    print (res)
