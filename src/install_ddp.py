import os
import urllib

from torchvision.datasets.utils import download_and_extract_archive

url = "" 
filename = "ddp.tar.gz"
md5 = "1b9587c2bdf7867e43fb9da345f395eb"

# WARNING: this will download and extract a 87GiB file, if not already present. Please save the file and avoid re-downloading it.
try:
    download_and_extract_archive(url=url, download_root=os.curdir, extract_root=None, filename=filename, md5=md5, remove_finished=False)
except urllib.error.HTTPError as e:
    print(e)
    print("Have you replaced the URL above with the one you got after registering?")