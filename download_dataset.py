import requests

url = 'https://crss.utdallas.edu/corpora/UT-Podcast/UT-Podcast.tar.gz'
# get the file name
if url.find('/'):
    file_name = url.rsplit('/', 1)[1]
r = requests.get(url, allow_redirects=True)
open(file_name, 'wb').write(r.content)