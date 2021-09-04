from pymongo import MongoClient
import branch

# Function to iterate through every discussion and extrtact user data
# dictionary structure - {username: [f1,f2,f3,f4,t,pa,pd,nd,ns,c]}
'''
f1 - number of questions
f2 - number of answers
f3 - number of upvotes
f4 - number of TA verification
t - total number of participated discussions
pa - number of agreement threads
pd - number of disagreement threads
nd - number of detailed threads (depth>4)(Ques + ans + 2 comments = depth 4)
ns - number of shallow threads (depth<=4)
c - number of comments   - not used in final set but can be inlcuded if needed
'''
def Create_data():
    client = MongoClient(port=27017)
    db=client.portal
    result = db.discussions.find({})
    Dict_main = {}
    users = {}
    # get all unique user ids
    for d in result:
        users[d['_id']] = []
        users[d['_id']].append(d['userid'])
        for i in d['answers']:
            result_ans = db.answers.find({"_id":i})
            for a in result_ans:
                users[d['_id']].append(a['userid'])
                for j in a['comments']:
                    result_com = db.comments.find({"_id":j})
                    for c in result_com:
                        users[d['_id']].append(c['userid'])
    # print(users)
    result = db.discussions.find({})
    for d in result:
        for i in d['answers']:
            result_ans = db.answers.find({"_id":i})
            for a in result_ans:
                Dict = {} 
                Dict[d['username']] = [1,0,0,0,1,0,0,0,0,0]
                for v in d['votes']:
                    if(str(v) in users[d['_id']]):
                        Dict[d['username']][2] += 1
                depth = 2
                if(a['username'] not in Dict):
                    Dict[a['username']] = [0,1,0,int(a['TA_verified']),1,0,0,0,0,0]
                else:
                    Dict[a['username']][1] += 1
                    Dict[a['username']][3] += int(a['TA_verified'])
                for v in a['votes']:
                    if(str(v) in users[d['_id']]):
                        Dict[a['username']][2] += 1
                coms = []
                for j in a['comments']:
                    result_com = db.comments.find({"_id":j})
                    for c in result_com:
                        coms.append(c)
                        depth += 1
                        if(c['username'] not in Dict):
                            Dict[c['username']] = [0,0,0,int(c['TA_verified']),1,0,0,0,0,1]
                        else:
                            Dict[c['username']][3] += int(c['TA_verified'])
                            Dict[c['username']][9] += 1
                        for v in c['votes']:
                            if(str(v) in users[d['_id']]):
                                Dict[c['username']][2] += 1
                result = branch.getResult(d,a,coms)
                for k in Dict:
                    if(result == 0):
                        Dict[k][5] = 1
                    else:
                        Dict[k][6] = 1
                    if(depth>4):
                        Dict[k][7] = 1  
                    else:
                        Dict[k][8] = 1
                Dict_main = merge(Dict_main,Dict)
    return Dict_main
            
    

# Every iteration merge new dictionary to current complete dictionary
def merge(d1,d2):
    for i in d2:
        if(i in d1):
            for j in range(len(d2[i])):
                d1[i][j]+=d2[i][j]
        else:
            d1[i]=d2[i]
    return d1

# weights
w = [1,1,1,1,1,1,1,1,1,1]
# Extract user data

user_data = Create_data()

#Calculate weighted average of the final features with(ratios) in the dictionary (Doesn't have number of comments)
Final_data = {}
for i in user_data:
    Final_data[i] = [0,0,0,0,0,0,0,0,0]
    Final_data[i][0]=user_data[i][0]
    Final_data[i][1]=user_data[i][1]
    Final_data[i][2]=user_data[i][2]
    Final_data[i][3]=user_data[i][3]
    Final_data[i][4]=(user_data[i][3]+1)/(2+user_data[i][3]+user_data[i][1])                           # f5 = (f4+1)/(f2+f4+2)   
    Final_data[i][5]=(user_data[i][2]+1)/(2+user_data[i][2]+user_data[i][1])                           # f6 = (f3+1)/(f2+f3+2)
    Final_data[i][6]=user_data[i][5]/user_data[i][4]                            # f7 = pa/t
    Final_data[i][7]=user_data[i][6]/user_data[i][4]                            # f8 = pd/t
    Final_data[i][8]=(user_data[i][7]+1)/(2+user_data[i][8]+user_data[i][1])                          # f9 = (nd+1)/(ns+nd+2)
    fwsum = 0                                                                   
    for j in range(len(Final_data[i])):
        fwsum+=(w[j]*Final_data[i][j])                                          # weighted sum
    Final_data[i].append(fwsum/sum(w))

# Final dictionary structure - {username: [f1,f2,f3,f4,f5,f6,f7,f8,f9,favg]}
'''
f1 - number of questions
f2 - number of answers
f3 - number of upvotes
f4 - number of TA verification
f5 - f4/f2
f6 - f3/f2
f7 - number of agreement threads/total number of participated discussions
f8 - number of disagreement threads/total number of participated discussions
f9 - number of detailed threads/number of shallow threads 
favg - weighted average of f1-9
'''     


# The data here is the initial extracted data without any ratios
sort_orders = sorted(Final_data.items(), key=lambda x: x[1][9], reverse=True)
for i in sort_orders:
	print(i[0], i[1])




