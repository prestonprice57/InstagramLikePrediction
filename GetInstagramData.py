import json
import requests
import indicoio
import numpy as np
from facepp import API as face_api
from instagram.client import InstagramAPI

client_id = "bec82b4b69cc435998eb2c9f82212fb4"
client_secret = "6f7cd017a78945afaffcd992840a8fe5"
access_token = "1147536024.bec82b4.fb48b565d9ad4fe09f64f63d64d4f664"
INDICO_API_KEY = '61cdd30af4bbdfe5a21b92689a872234'

FACE_API_KEY = '57a5f94a17c3bbf07823b6e4d06dde10'
FACE_API_SECRET = '1le1_mXDShAHhwG5OPjjjbpqpT14fEKG'

api = InstagramAPI(access_token=access_token, client_secret=client_secret)
face = face_api(FACE_API_KEY, FACE_API_SECRET)
indicoio.config.api_key = INDICO_API_KEY

INPUT_FILE = 'InstagramUsersx100000.txt'

users = []
f = open(INPUT_FILE, 'r')
for line in f:
    new_line, end_line = line.split('\n')
    users.append(new_line)
f.close()

np.random.seed(1024)
indices = np.random.permutation(len(users))

startIndex = 420
endIndex = 430
while endIndex < len(indices): 
    jsonFile = {}
    with open('data.json') as infile:
        jsonFile = json.load(infile)

    for i in range(startIndex, endIndex):
        new_user = users[indices[i]]
        print new_user


        isPrivate = True
        try: 
            isPrivate = api.user_relationship(user_id=new_user).target_user_is_private
        except:
            print "This didn't work"

        if isPrivate == False:
            url = 'https://api.instagram.com/v1/users/%s/?access_token=%s' % (new_user, access_token)
            resp = requests.get(url=url)
            data = resp.json()
            follows = data['data']['counts']['follows']
            followers = data['data']['counts']['followed_by']

            recent_media, next = api.user_recent_media(user_id=new_user, count=4)
            for media in recent_media:
                image_url = media.get_standard_resolution_url()
                faceNum = 0
                try:
                    faceNum = len(face.detection.detect(url=image_url)['face'])
                except:
                    pass

                if media.type != 'video' and 1 <= faceNum <= 4:
                    day = media.created_time.weekday()
                    hour = str(media.created_time.hour) + ':' + str(media.created_time.minute)
                    likes = media.like_count
                    hashtags = len(media.tags)

                    captionSentiment = 0.5
                    if media.caption != None:
                        caption = media.caption.text.replace('\n', ' ').replace('\r', ' ').encode('utf-8')
                        captionSentiment = indicoio.sentiment(caption)

                    fer = indicoio.fer(image_url)

                    new_post = {}
                    new_post['happy'] = fer['Happy']
                    new_post['sad'] = fer['Sad']
                    new_post['angry'] = fer['Angry']
                    new_post['fear'] = fer['Fear']
                    new_post['surprise'] = fer['Surprise']
                    new_post['neutral'] = fer['Neutral']
                    new_post['day'] = day
                    new_post['hour'] = hour
                    new_post['likes'] = likes
                    new_post['follows'] = follows
                    new_post['followers'] = followers
                    new_post['hashtags'] = hashtags
                    new_post['captionSentiment'] = captionSentiment
                    if followers > 0:
                        new_post['likeRatio'] = float(likes) / followers
                    else:
                        new_post['likeRatio'] = float(likes)
                    if follows > 0:
                        new_post['followerRatio'] = float(followers) / follows
                    else:
                        new_post['followerRatio'] = float(followers)
                    jsonFile['data']['posts'].append(new_post)

    f5 = open('NumPosts.txt', 'a')
    f5.write("%s\n" % endIndex)
    f5.close()

    print(endIndex)

    startIndex += 10
    endIndex += 10

    with open('data.json', 'w') as outfile:
        json.dump(jsonFile, outfile, sort_keys=True, indent=4)
