import os
import urllib

from torchvision.datasets.utils import download_and_extract_archive

files = [
    {
        'filename' : 'sst2_lo.tar.gz',
        'url': '',
        'md5': '205414dd0217065dcebe2d8adc1794a3'
    },
    {
        'filename' : 'sst2_hi.tar.gz',
        'url': '',
        'md5': 'd285958529fcad486994347478feccd2'
    },
    {
        'filename' : 'sst2_inf.tar.gz',
        'url': '',
        'md5': '7dca44b191a0055edbde5c486a8fc671'
    }
]

# WARNING: this will download and extract three ~80GiB files, if not already present. Please save the files and avoid re-downloading them.
try:
    for f in files:
        url, filename, md5 = f['url'], f['filename'], f['md5']
        print(f"Downloading and extracting {filename}...")
        download_and_extract_archive(url=url, download_root=os.curdir, extract_root=None, filename=filename, md5=md5, remove_finished=False)
except urllib.error.HTTPError as e:
    print(e)
    print("Have you replaced the URLs above with the one you got after registering?")