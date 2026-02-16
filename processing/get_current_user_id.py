import httpx

API_URL = "https://fancamai.com/api/result/get_current_user_id"

def call_get_current_user_id_api(user_token):

    headers = {
        "Authorization": f"Bearer {user_token}"
    }

    with httpx.Client() as client:
        response = client.get(API_URL, headers=headers)

    if response.status_code != 200:
        raise Exception("Failed get current user id")

    return response.json()["current_user_id"]
