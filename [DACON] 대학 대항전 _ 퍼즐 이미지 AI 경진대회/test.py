import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import pandas as pd

train_df = pd.read_csv('/home/aiuser/kyunghoney/data/Puzzle/train.csv')
test_df = pd.read_csv('/home/aiuser/kyunghoney/data/Puzzle/test.csv')
train_name = os.listdir('/home/aiuser/kyunghoney/data/Puzzle/test')

# n번째 이미지 넣으면 상, 하, 좌, 우 자른 array 리턴해줌
def image_cut_side(image, NUM, SIZE, pixel):
    p = pixel//4
    x = NUM%4
    y = NUM // 4
    image_right = image[p*y:p*(y+1),p*(x+1)-SIZE:p*(x+1),:]
    image_left = image[p*y:p*(y+1),p*x:p*x+SIZE,:]
    image_up = image[p*y:p*y+SIZE,p*x:p*(x+1),:]
    image_down = image[p*(y+1)-SIZE:p*(y+1),p*x:p*(x+1),:]
    # print(image_up.shape)
    # print(image_down.shape)
    # print(image_left.shape)
    # print(image_right.shape)
    return image_up, image_down, image_left, image_right

# 두 array 값을 빼서 제곱한 후 더한 값 리턴해줌
def image_SD(l1,l2):
    r = (l1-l2)**2
    r = r*(np.pi/2)
    r = np.tan(r)
    r = r.reshape(-1).sum()
    # if r == 0:
    #     print(l1,l2)
    return r

# 상, 하, 좌, 우 이미지 유사도를 체크해 줌(right 함수는 넣은 이미지의 오른쪽에 올 때 로스를 순서대로 보여줌 - answer에 있는것은 제외하는데 이때 제외된 값의 로스는 100)
def Find_right(image,NUM,answer,pixel):
    rt_r = []
    for i in range(16):
        if i not in answer and i != NUM:
            _,_,_,r = image_cut_side(image,NUM,1,pixel)
            _,_,l,_ = image_cut_side(image,i,1,pixel)
            rt_r.append(image_SD(r,l))
        else:
            rt_r.append(1000000000)
    #print('r',rt_r,answer)
    return rt_r

def Find_left(image,NUM,answer,pixel):
    rt_l = []
    for i in range(16):
        if i not in answer and i != NUM:
            _,_,l,_ = image_cut_side(image,NUM,1,pixel)
            _,_,_,r = image_cut_side(image,i,1,pixel)
            rt_l.append(image_SD(r,l))
        else:
            rt_l.append(1000000000)
    #print('l',rt_l)
    return rt_l

def Find_up(image,NUM,answer,pixel):
    rt_u = []
    for i in range(16):
        if i not in answer and i != NUM:
            u,_,_,_ = image_cut_side(image,NUM,1,pixel)
            _,d,_,_ = image_cut_side(image,i,1,pixel)
            rt_u.append(image_SD(d,u))
        else:
            rt_u.append(1000000000)
    #print('u',rt_u)
    return rt_u

def Find_down(image,NUM,answer,pixel):
    rt_d = []
    for i in range(16):
        if i not in answer and i != NUM:
            _,d,_,_ = image_cut_side(image,NUM,1,pixel)
            u,_,_,_ = image_cut_side(image,i,1,pixel)
            rt_d.append(image_SD(d,u))
        else:
            rt_d.append(1000000000)
    #print('d',rt_d)
    return rt_d

# ANS_F에 첫 키(왼쪽 위) 하나 넣어주면 나머지 넣어주는 함수(정답과 loss를 리턴해줌)-오른쪽 먼저
def Find_ans0(image,ANS,pixel):
    loss = 0
    ANS_F = ANS.copy()
    for i in range(1,5):
        for j in range(1,5):
            if ANS_F[i][j] == -1:
                sum = np.array([0 for i in range(16)])
                if ANS_F[i-1][j] >= 0:
                    sum = sum + np.array(Find_down(image,ANS_F[i-1][j],ANS_F,pixel))
                if ANS_F[i+1][j] >= 0:
                    sum = sum + np.array(Find_up(image,ANS_F[i+1][j],ANS_F,pixel))
                if ANS_F[i][j-1] >= 0:
                    sum = sum + np.array(Find_right(image,ANS_F[i][j-1],ANS_F,pixel))
                if ANS_F[i][j+1] >= 0:
                    sum = sum + np.array(Find_left(image,ANS_F[i][j+1],ANS_F,pixel))
                ANS_F[i][j] = int(np.where(sum == min(sum))[0][0])
                if len(np.where(sum == min(sum))[0])>1:
                    print('Find_ans_Fwe0', min(sum), np.where(sum == min(sum))[0])
                loss += min(sum)
    return loss

# ANS_F에 첫 키(왼쪽 위) 하나 넣어주면 나머지 넣어주는 함수(정답과 loss를 리턴해줌)-아래 먼저
def Find_ans1(image,ANS,pixel):
    loss = 0
    ANS_F = ANS.copy()
    for j in range(1,5):
        for i in range(1,5):
            if ANS_F[i][j] == -1:
                sum = np.array([0 for i in range(16)])
                if ANS_F[i-1][j] >= 0:
                    sum = sum + np.array(Find_down(image,ANS_F[i-1][j],ANS_F,pixel))
                if ANS_F[i+1][j] >= 0:
                    sum = sum + np.array(Find_up(image,ANS_F[i+1][j],ANS_F,pixel))
                if ANS_F[i][j-1] >= 0:
                    sum = sum + np.array(Find_right(image,ANS_F[i][j-1],ANS_F,pixel))
                if ANS_F[i][j+1] >= 0:
                    sum = sum + np.array(Find_left(image,ANS_F[i][j+1],ANS_F,pixel))
                ANS_F[i][j] = int(np.where(sum == min(sum))[0][0])
                if len(np.where(sum == min(sum))[0])>1:
                    print('Find_ans_Fwe0', min(sum), np.where(sum == min(sum))[0])
                loss += min(sum)
    return loss

# ans_H의 i, j번째에 들어갈 수 있는 이미지 번호를 보내줌(min_loss)보다 작으면 보내주고 그런거 없으면 최솟값 보내주고
def have_loss(image,ANS_H,i,j,min_loss,pixel):
    p = (pixel/4)*3
    n=0
    sum = np.array([0 for i in range(16)])
    if ANS_H[i-1][j] >= 0:
        sum = sum + np.array(Find_down(image,ANS_H[i-1][j],ANS_H,pixel))/p
        n+=1
    if ANS_H[i+1][j] >= 0:
        sum = sum + np.array(Find_up(image,ANS_H[i+1][j],ANS_H,pixel))/p
        n+=1
    if ANS_H[i][j-1] >= 0:
        sum = sum + np.array(Find_right(image,ANS_H[i][j-1],ANS_H,pixel))/p
        n+=1
    if ANS_H[i][j+1] >= 0:
        sum = sum + np.array(Find_left(image,ANS_H[i][j+1],ANS_H,pixel))/p
        n+=1
    sum = sum/n
    if len(np.where(sum < min_loss)[0])>0:
        return np.where(sum < min_loss)[0]
    else:
        return np.where(sum == min(sum))[0]


# ANS에 첫 키(왼쪽 위) 하나 넣어주면 나머지 넣어주는 함수(정답과 loss를 리턴해줌)-오른쪽 먼저 : 이때 min_loss보다 작은건 모두 보내줌
def Find_ans0_while(image,ANS,min_loss,pixel):
    loss = 0
    for i in range(1,5):
        for j in range(1,5):
            check_loss = 100000
            ansij = -1
            if ANS[i][j] == -1:
                haveminloss = have_loss(image, ANS, i,j, min_loss,pixel)
                for h in haveminloss:
                    ANS[i][j] = h
                    ans_copy = ANS.copy()
                    find_loss = Find_ans0(image,ans_copy,pixel)
                    if find_loss < check_loss:
                        check_loss = find_loss
                        ansij = h
                ANS[i][j] = ansij
            loss += check_loss
    ANS = ANS[1:5,1:5].reshape(-1)
    #print(loss) ###############################################
    return ANS, loss

# ANS에 첫 키(왼쪽 위) 하나 넣어주면 나머지 넣어주는 함수(정답과 loss를 리턴해줌)-오른쪽 먼저 : 이때 min_loss보다 작은건 모두 보내줌
def Find_ans1_while(image,ANS,min_loss,pixel):
    loss = 0
    for j in range(1,5):
        for i in range(1,5):
            check_loss = 100000
            ansij = -1
            if ANS[i][j] == -1:
                haveminloss = have_loss(image, ANS, i,j, min_loss,pixel)
                for h in haveminloss:
                    ANS[i][j] = h
                    ans_copy = ANS.copy()
                    find_loss = Find_ans0(image,ans_copy,pixel)
                    if find_loss < check_loss:
                        check_loss = find_loss
                        ansij = h
                ANS[i][j] = ansij
            loss += check_loss
    ANS = ANS[1:5,1:5].reshape(-1)
    return ANS, loss
# 데이콘 답으로 바꾸기
def cha(l):
    r = np.zeros(16,dtype=int)
    #print(l)
    for i in range(16):
        # print(np.where(l==i)[0])
        r[i] = np.where(l==i)[0]
    return r+1

# 로스 loss_min이하인것 다 해보기

from tqdm import tqdm
num = 2
pixel = 512
loss_min=0.0089
acc = []


submit = pd.read_csv('/home/aiuser/kyunghoney/data/Puzzle/sample_submission.csv')
for i in tqdm(range(submit.shape[0])):
    img = Image.open('/home/aiuser/kyunghoney/data/Puzzle/test/'+submit.ID[i]+".jpg")
    img = img.resize((pixel,pixel))
    img = np.array(img)
    image_scaled = img / 256

    answer = np.ones(16)
    loss = 100000000
    for a in range(16):
        ans = -1 * np.ones((6,6),dtype='int')
        ans[1][1] = a
        find_ans0, find_loss0 = Find_ans0_while(image_scaled, ans, loss_min, pixel)
        ans = -1 * np.ones((6,6),dtype='int')
        ans[1][1] = a
        find_ans1, find_loss1 = Find_ans1_while(image_scaled, ans, loss_min, pixel)
        
        if find_loss1 < find_loss0:
            find_loss = find_loss1
            find_ans = find_ans1
        else:
            find_loss = find_loss0
            find_ans = find_ans0

        if find_loss < loss:
            answer = find_ans
            loss = find_loss

    answer = cha(answer)
    submit.iloc[i, 1:] = answer
    
submit.to_csv("./savewhile.csv", index=False)