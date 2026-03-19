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

    timeout = Timeout(connect=10.0, read=25.0)
    max_attempts = 2
    retry_delay = 1

    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.post(API_URL, data=data, headers=headers)

            if response.status_code != 200:
                raise Exception(f"Failed save result: {response.status_code}")

            return response.json()

        except httpx.ReadTimeout:
            if attempt == max_attempts:
                raise Exception("All attempts to save result timed out")
            print(f"Attempt {attempt} timed out. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)

    raise Exception("call_save_result_api failed unexpectedly")