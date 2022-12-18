import os
import urllib

from torchvision.datasets.utils import download_and_extract_archive

files = [
    {
        'filename' : 'sst2_lo.tar.gz',
        'url': 'https://membershipinference.blob.core.windows.net/mico/sst2_lo.tar.gz?si=sst2&spr=https&sv=2021-06-08&sr=b&sig=amrEufHCVfF51mWKxzGsa1gsQCjqwwZnDWiYEDt6x8w%3D',
        'md5': '205414dd0217065dcebe2d8adc1794a3'
    },
    {
        'filename' : 'sst2_hi.tar.gz',
        'url': 'https://membershipinference.blob.core.windows.net/mico/sst2_hi.tar.gz?si=sst2&spr=https&sv=2021-06-08&sr=b&sig=h0SFb4W6tFo4n%2FsSJkzMCdUHNH8enNNc6NV%2BaFuSDGE%3D',
        'md5': 'd285958529fcad486994347478feccd2'
    },
    {
        'filename' : 'sst2_inf.tar.gz',
        'url': 'https://membershipinference.blob.core.windows.net/mico/sst2_inf.tar.gz?si=sst2&spr=https&sv=2021-06-08&sr=b&sig=nn2NVJathV1exL%2BZEDc5%2F08YkROVgVEhDWmCfAhBi9I%3D',
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