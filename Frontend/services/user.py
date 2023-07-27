import requests

BASE_API_URL = "http://backend:8000/"

def get_all_user_data():
    try:
        # Make the HTTP GET request
        response = requests.get(BASE_API_URL + "user_data")

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # If the response contains JSON data, you can access it using response.json()
            data = response.json()
            return data
        else:
            # If the request was not successful, raise an exception with the status code and reason
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that may occur during the request
        print(f"Error occurred: {e}")

def get_user_data_by_name(firstname:str, lastname:str):
    try:
        # Make the HTTP GET request
        response = requests.get(BASE_API_URL + f"get_user_info?name={firstname}&lastname={lastname}")

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # If the response contains JSON data, you can access it using response.json()
            data = response.json()
            return data
        else:
            # If the request was not successful, raise an exception with the status code and reason
            response.raise_for_status()
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that may occur during the request
        print(f"Error occurred: {e}")