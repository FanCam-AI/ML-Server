import httpx

API_URL = "https://fancamai.com/result/save_result"

def call_save_result_api(output_path, file_type, user_token):
    data = {
        "title": "",
        "file_path": output_path,
        "file_type": file_type
    }

    headers = {
        "Authorization": f"Bearer {user_token}"
    }

    with httpx.Client() as client:
        response = client.post(API_URL, data=data, headers=headers)

    if response.status_code != 200:
        raise Exception("Failed save result")

    return response.json()
