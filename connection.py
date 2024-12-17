import http.client
import json
from datetime import datetime, time, timedelta
import time as sleep_time
from data.csv_data import update_errorlog


USERNAME = "username"
PASSWORD = "password"


# ========== Globals =======
BASE_URL = "api-capital.backend-capital.com"  # Replace with the actual API endpoint
DEMO_BASE_URL = "demo-api-capital.backend-capital.com"  # Replace with the actual API endpoint
EXTENDED_URL = "/api/v1/"
API_KEY = "api-key"  # Replace with your actual API key
CST_TOKEN = ""
X_SECURITY_TOKEN = ""

def create_new_session():
    global CST_TOKEN, X_SECURITY_TOKEN  # Declare the tokens as global

    conn = http.client.HTTPSConnection(BASE_URL)
    payload = json.dumps({
        "identifier": USERNAME,
        "password": PASSWORD
    })
    headers = {
        'X-CAP-API-KEY': API_KEY,
        'Content-Type': 'application/json'
    }
    conn.request("POST", f"{EXTENDED_URL}session", payload, headers)
    res = conn.getresponse()

    # Fetch the tokens
    X_SECURITY_TOKEN = res.getheader('X-SECURITY-TOKEN')
    CST_TOKEN = res.getheader('CST')
    data = res.read()


def make_request(method, endpoint, payload=None, retry=True):
    retry_counter = 0
    while True:
        try:
            conn = http.client.HTTPSConnection(DEMO_BASE_URL)
            if payload is not None:
                payload = json.dumps(payload) if isinstance(payload, dict) else payload
            headers = {
                'X-SECURITY-TOKEN': X_SECURITY_TOKEN,
                'CST': CST_TOKEN,
                'Content-Type': 'application/json'
            }
            conn.request(method, endpoint, payload, headers)
            res = conn.getresponse()
            data = res.read().decode("utf-8")
            if res.status == 200:
                return json.loads(data)
            else:
                if retry:
                    print("Error response:")
                    print("Status Code:", res.status)
                    print("Reason:", res.reason)
                    print("Response Data:", data)
                    now = datetime.now()  # Get the current date and time
                    retry_datetime = now.strftime("%d-%m-%Y %H:%M:%S")
                    retry_counter += 1
                    print(retry_datetime, "Retrying in 15 seconds..., Retry:", retry_counter, endpoint)
                    update_errorlog(retry_datetime, endpoint, retry_counter)
                    if retry_counter > 8:
                        return None
                    else:
                        sleep_time.sleep(15)
                else:
                    return None
        except Exception as e:
            print(f"Error during request: {str(e)}")
            if retry:
                print("Retrying in 15 seconds...")
                retry_counter += 1
                print("Retry:", retry_counter)
                if retry_counter > 8:
                    return None
                else:
                    sleep_time.sleep(15)
            else:
                return None

def fetch_us100_snapshot():
    dict_us100_market = make_request("GET", f"{EXTENDED_URL}markets/US100")
    dict_us100_snapshot = dict_us100_market.get('snapshot', {})
    return dict_us100_snapshot

def fetch_us100_history():
    now = datetime.now()  # Get the current date and time
    start_datetime = "2024-12-11T12:00:00"
    end_datetime = "2024-12-11T13:00:00"
    print(f"Dynamic Query: {EXTENDED_URL}prices/US100?resolution=MINUTE&max=50&from={start_datetime}&to={end_datetime}")

    # Format the request dynamically
    dict_us100_history = make_request(
        "GET",
        f"{EXTENDED_URL}prices/US100?resolution=MINUTE&max=50&from={start_datetime}&to={end_datetime}"
    )
    # dict_us100_history = make_request("GET",f"{EXTENDED_URL}prices/US100?resolution=MINUTE&max=50&from=2024-10-15T12:00:00&to=2024-10-15T15:00:00")
    return dict_us100_history

def account_details():
    return make_request("GET", f"{EXTENDED_URL}accounts")


def create_position(payload):
    return make_request("POST", f"{EXTENDED_URL}positions", payload=payload)


def close_position(dealId):
    return make_request("DELETE", f"{EXTENDED_URL}positions/{dealId}")

def active_account():
    conn = http.client.HTTPSConnection(DEMO_BASE_URL)
    payload = json.dumps({
        "accountId": "243927243034023070"
    })
    headers = {
        'X-SECURITY-TOKEN': X_SECURITY_TOKEN,
        'CST': CST_TOKEN,
        'Content-Type': 'application/json'
    }
    conn.request("PUT", "/api/v1/session", payload, headers)
    res = conn.getresponse()
    data = res.read()

def get_open_position():
    active_account()
    conn = http.client.HTTPSConnection(DEMO_BASE_URL)
    payload = ''
    headers = {
        'X-SECURITY-TOKEN': X_SECURITY_TOKEN,
        'CST': CST_TOKEN
    }
    conn.request("GET", "/api/v1/positions", payload, headers)
    res = conn.getresponse()
    while res.status != 200:
        time.sleep(5)
        conn.request("GET", "/api/v1/positions", payload, headers)
        res = conn.getresponse()
    data = res.read()
    positions_data = json.loads(data.decode("utf-8"))
    if "positions" in positions_data and len(positions_data["positions"]) > 0:
        deal_id = positions_data["positions"][0]["position"]["dealId"]
        return deal_id
    else:
        return ("No open positions at this moment.")

