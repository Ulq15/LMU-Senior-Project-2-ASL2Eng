import os
import os.path as path
import json


filenames = set(os.listdir(path.join(os.pardir, 'data/videos')))
jsonFile = path.join(os.path.dirname(__file__), 'WLASL_v0.3.json')
content = json.load(open(jsonFile))

missing_ids = []

for entry in content:
    instances = entry['instances']

    for inst in instances:
        video_id = inst['video_id']
        if video_id + '.mp4' not in filenames:
            missing_ids.append(video_id)


with open(path.join(os.path.dirname(__file__),'data/missing.txt'), 'w') as f:
    f.write('\n'.join(missing_ids))

