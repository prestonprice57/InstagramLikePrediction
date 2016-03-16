from instagram.client import InstagramAPI
import indicoio
from collections import Counter

client_id = "da4a3124488840f582305f64b3557548"
client_secret = "e7dce52052b842fb909ac0c24b467afb"
access_token = "1147536024.bec82b4.fb48b565d9ad4fe09f64f63d64d4f664"
INDICO_API_KEY = '61cdd30af4bbdfe5a21b92689a872234'

api = InstagramAPI(access_token=access_token, client_secret=client_secret)
indicoio.config.api_key = INDICO_API_KEY

following, next = api.user_follows(user_id="self", count=10)
#print next
#temp, max_tag=next.split('max_tag_id=')
#max_tag = str(max_tag)

usernames = []
for f in following:
	isPrivate = api.user_relationship(user_id=f.id).target_user_is_private
	if isPrivate == False:
		usernames.append(f.id)
print len(usernames)

new_usernames = usernames[:]
for user in usernames:
	length = len(usernames)
	if length > 100000:
		break
	else:
		isPrivate = api.user_relationship(user_id=user).target_user_is_private
		if isPrivate == False:
			following, next = api.user_follows(user_id=user, count=100)
			for f in following:
				isPrivate2 = api.user_relationship(user_id=f.id).target_user_is_private
				if f.id not in usernames and isPrivate2 == False:
					usernames.append(f.id)

print len(usernames)


f = open('InstagramUsers.txt', 'w')
for user in usernames:
	f.write("%s\n" % user)

f.close()

'''
counter = 1
while next and counter < 4:
	more_following, next = api.user_follows(user_id="self", max_tag_id=max_tag)
	temp, max_tag = next.split('max_tag_id=')
	max_tag=str(max_tag)
	for f in more_following:
		usernames.append(f.username)
	counter+=1
'''




