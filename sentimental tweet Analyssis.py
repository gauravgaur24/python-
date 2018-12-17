import imp 
import re # regular expression 
import tweepy #access to tweet app
import matplotlib.pyplot as plt
from tweepy import OAuthHandler #authenication 
from textblob import TextBlob #text/tweet parse

ck="Jm6WDXZuiwlwUnT9mFbPdSpcg"
cs="Wp4Gbf74R6MZWjXvj0ifKrCobwWbchhn53Mv8L7VMLbIUi6Wnd"
at="919434545924935681-wFjVVTbs0pmyB2VwSoj4VwGb7tBYCyr"
ats ="RaPfxU0rSMjkS3MSI9N0ztXu4I2iLMecXg79OerNHw4Ly"
try:
    au = OAuthHandler(ck, cs)
    au.set_access_token(at, ats)
    ob = tweepy.API(au) #login 
    print("Authentication successful")

except:
    print("authentication Fails")
   

def cleantweet(a):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ",a).split())  #cleaning of text
    
def sentiment(t):
    analysis = TextBlob(t)   #to check the polarity of each word 
     
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'

#user will enter the subject and the count of tweet which has to analyze 
user =str(input("enter the subject you want to analyze--"))
c=int(input("enter the no of tweets to be analyze--"))

leader=[]

out = ob.search(q=user,count=c)
for o in out:
    k=cleantweet(o.text) 
    #print(k)
    leader.append(k)

p=[]
n=[]
neg=[]
o=[]
for l in leader:
    
    o=sentiment(l)
    if o=='positive':
        p.append(o)
    elif o=='neutral':
        n.append(o)
    elif o=='negative':
        neg.append(o)
    
#length of each sentiment 
plen=len(p)
nlen=len(n)
neg_len=len(neg)
total=plen+nlen+neg_len

#percentage of the tweet 
print("positive sentimentals are==",plen*100/total)
print("neutral sentimentals are==",nlen*100/total)
print("negative sentimentals are==",neg_len*100/total)
pos=plen*100/total
neg=neg_len*100/total
neut=nlen*100/total

#Graphical representation of the sentiment on pie chart 
colors = ['green', 'blue', 'red']
sizes = [pos, neg, neut]
labels = 'Positive', 'Negative', 'Neutral'
plt.pie(
   x=sizes,
   shadow=False,
   colors=colors,
   labels=labels,
   startangle=90
)

plt.show()





   


#' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ",o.text).split())


