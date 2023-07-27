import requests

BASE_API_URL = "http://localhost:8000/"

def download_pdf_from_api(st):
    try:
        # Send a GET request to the API to fetch the PDF data
        response = requests.get(BASE_API_URL + "generate_decision_plot")

        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Set content type as PDF to enable download
            # st.header("Download PDF:")
            # st.download_button(label="Download PDF", data=response.content, file_name="user_report.pdf", mime="application/pdf")
            st.write("HELLO")
        else:
            st.error(f"Failed to fetch PDF from API. Status code: {response.status_code}")

    except requests.exceptions.RequestException as e:
        st.error(f"Error occurred while fetching PDF from API: {e}")