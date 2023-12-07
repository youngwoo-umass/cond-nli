import io
import zipfile

import requests


def download_and_unpack_zip(url, extract_to):
    """
    Download a zip file from the given URL and unpack it to the specified directory.

    Parameters:
    url (str): URL of the zip file to download.
    extract_to (str): Directory where to extract the contents of the zip file. Defaults to current directory.
    """
    # Send a GET request to the URL
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Create a ZipFile object from the content of the response
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        # Extract all the contents of the zip file
        zip_file.extractall(extract_to)
        print(f"Zip file from '{url}' extracted to '{extract_to}'")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def download_file(url, save_path):
    # Send a GET request to the specified URL
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a file with the specified filename in binary write mode
        with open(src/utils/download_util.py, 'wb') as file:
            # Write the content of the response to the file
            for chunk in response.iter_content(chunk_size=128):
                file.write(chunk)
        print(f"File downloaded and saved as {save_path}")
    else:
        print(f"Failed to download file: status code {response.status_code}")
