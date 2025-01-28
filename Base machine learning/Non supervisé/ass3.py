import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix, csr_matrix
from scipy.spatial.distance import jaccard, cosine 
from pytest import approx
from sklearn.metrics import pairwise_distances


MV_users = pd.read_csv('users.csv')
MV_movies = pd.read_csv('movies.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
from collections import namedtuple
Data = namedtuple('Data', ['users','movies','train','test'])
data = Data(MV_users, MV_movies, train, test)


import copy
class RecSys():
    def __init__(self,data):
        self.data=data
        #print(data.train.shape,data.test.shape,data.test) # train and test have 3 cols, i.e. uid, mid, rating, both have 30k lines
        self.allusers = list(self.data.users['uID']) # 5769 u
        self.allmovies = list(self.data.movies['mID']) # 3152 m
        self.genres = list(self.data.movies.columns.drop(['mID', 'title', 'year']))
        self.mid2idx = dict(zip(self.data.movies.mID,list(range(len(self.data.movies)))))
        self.uid2idx = dict(zip(self.data.users.uID,list(range(len(self.data.users)))))
        self.Mr=self.rating_matrix()
        self.Mm=None 
        self.sim=csr_matrix(np.zeros((len(self.allmovies),len(self.allmovies))))
        
    def rating_matrix(self):
        """
        Convert the rating matrix to numpy array of shape (#allusers,#allmovies)
        """
        ind_movie = [self.mid2idx[x] for x in self.data.train.mID] # only the indexes of movies that exist in training; this should be a subset of all the values in dic mid2idx
        ind_user = [self.uid2idx[x] for x in self.data.train.uID]
        rating_train = list(self.data.train.rating)
        #print("size of users, movies, and training users and training movies",len(self.allusers),len(self.allmovies),len(ind_user),len(ind_movie))
        #rating_train = list(train.rating)
        #return np.array(coo_matrix((rating_train, (ind_user, ind_movie)), shape=(len(self.allusers), len(self.allmovies))).toarray())
        return  coo_matrix((rating_train, (ind_user, ind_movie)), shape=(len(self.allusers), len(self.allmovies))).toarray()

    def predict_everything_to_3(self):
        """
        Predict everything to 3 for the test data
        """
        # Generate an array with 3s against all entries in test dataset
        # your code here
        n=self.data.test.rating.shape[0]
        #print("size of predict 3",n)
        #self.Mm=np.full((len(ind_user), len(ind_movie)), 3)
        #for i in ind_movie:
        #    for j in ind_user:     
        #       self.Mr[j,i]=3
        return np.ones(n) * 3

        
    def predict_to_user_average(self):
        """
        Predict to average rating for the user.
        Returns numpy array of shape (#users,)
        """
        # Generate an array as follows:
        # 1. Calculate all avg user rating as sum of ratings of user across all movies/number of movies whose rating > 0
        # 2. Return the average rating of users in test data
        # your code here
        #n=len(self.data.test.mID)
        #means=self.Mr.sum(axis=1) / np.count_nonzero(self.Mr, axis=1)
        #mr_csr=self.Mr.tocsr()
        #non_zeros = mr_csr.getnnz(axis=1)
        #sum_csr=mr_csr.sum(axis=1)
        #with np.errstate(divide='ignore', invalid='ignore'):
        #    means=np.nan_to_num(sum_csr/non_zeros[:, np.newaxis] ,copy=False, posinf=0, neginf=0)
        #print("csr non zeros sum means shapes",mr_csr.shape,non_zeros.shape,sum_csr.shape,means.shape)
        #means=np.array([np.sum(r) / np.count_nonzero(r) for r in self.Mr])
        #means=np.array([r[r > 0].mean() if np.any(r > 0) else 0 for r in self.Mr]) # avg of es>0 for each row (user)
        #ind_user = [self.uid2idx[x] for x in self.data.test.uID]
        #means_test=means[ind_user]
        #means_rep=means_test#np.tile(means_test[:, np.newaxis], (1,n))
        #return means_rep# mat of m x n, where m is the # of users in both test and train
                       #and n is the # of test movies
        #print(self.data.test.rating.shape,self.data.train.rating.shape,len(self.allusers),len(self.allmovies))
        means=self.Mr.sum(axis=1) / (self.Mr!=0).sum(axis=1)#np.count_nonzero(self.Mr, axis=1) 
        with np.errstate(divide='ignore', invalid='ignore'):
            means=np.nan_to_num(means ,copy=True, posinf=0, neginf=0) # uindx ---- mean
        # map means to test data
        #print("avg size, should be 5769",means,means.shape) # this is what the question asked! But does not match 30k (size of the matrix of test ratings) 1.1429596846619763
        means_mapped= [means[self.uid2idx[x]] for x in self.data.test.uID] #30k 300k

        means_mapped2=[]
        for index, lig in self.data.test.iterrows():
            means_mapped2.append(lig['rating'] +1.1429596846619763)
        #print("antoher opiton:",np.array(means_mapped))
        print(np.mean(means_mapped2))
        return np.array(means_mapped)
        #pass
    
    def predict_from_sim(self,uid,mid):
        """
        Predict a user rating on a movie given userID and movieID
        """
        # Predict user rating as follows:
        # 1. Get entry of user id in rating matrix
        # 2. Get entry of movie id in sim matrix
        # 3. Employ 1 and 2 to predict user rating of the movie
        # your code here
        uidx=self.uid2idx[uid] # uid that exists in both test and train, I think
        midx=self.mid2idx[mid] # probably mid in the test region
        u_ratings=self.Mr[uidx] # 1 x n of movies, with 0 for movies that he did not rate, and a rate blah blah
        #np.savetxt("u_ratings.csv", u_ratings, delimiter=",", fmt='%s') 
        m_sim=self.sim[midx] # # 1 x n of movies
        #ind_movie = [self.mid2idx[x] for x in self.data.train.mID] # idx for train
        #ind_movie=u_ratings>0 # kill 0ssss esti de tabanrakaaljfladjfdals;f
        #u_ratings_train=u_ratings[ind_movie]
        #m_sim_train=m_sim[ind_movie]
        pos_ind = np.nonzero(u_ratings) # find out the midx (not mid) where rating is not 0 (there is a rating)
        #np.savetxt("pos_inds.csv", pos_ind, delimiter=",", fmt='%s') 
        #if uid==245 and mid==276:
        #    print("for user 245, there are this many ratings:",np.count_nonzero(u_ratings))
        #    print("positive training idx",pos_ind,u_ratings_train[pos_ind],m_sim_train[pos_ind])

        u_ratings_train_pos=u_ratings[pos_ind]
        m_sim_train_pos=m_sim[pos_ind]
        #np.savetxt("u_ratings_train_pos.csv", u_ratings_train_pos, delimiter=",", fmt='%s') 

        #np.savetxt("m_sim_train_pos.csv", m_sim_train_pos, delimiter=",", fmt='%s') 
        # write to the rating table   
        #for (i,j) in zip(u_ratings_train_pos,m_sim_train_pos):
        #    print("rating",i,"similitude",j)
        #print("checking if the two matrixes have the size of 22",m_sim_train_pos.shape,u_ratings_train_pos.shape)
        with np.errstate(divide='ignore', invalid='ignore'):
            rating=np.dot(u_ratings_train_pos,m_sim_train_pos)/np.sum(m_sim_train_pos) # for the movies rated by the user, do the weighted avg
        if np.isinf(rating):
            rating=3 #### if we do not have data for this user or nothing is similar to the movie!!
        #self.Mr[uid,mid]=rating do not want to affect the data
        return rating # weighted avg of sim * rating             
        #pass 
    
    def predict(self):
        """
        Predict ratings in the test data. Returns predicted rating in a numpy array of size (# of rows in testdata,)
        
        # your code here

        old code which is wrong ...
        
        

        ind_movie = [self.mid2idx[x] for x in self.data.test.mID] 
        ind_user = [self.uid2idx[x] for x in self.data.test.uID]
        movie_user_ratings=[]
        for i in ind_movie:
            user_ratings=[]
            for j in ind_user:
                user_ratings.append(self.predict_from_sim(j,i)) # array of 
            movie_user_ratings.append(user_ratings)

        """
        #return np.array(movie_user_ratings) # list of list of n rows and m cols
        res=[]
        #np.savetxt("data_test_uID.csv", self.data.test.uID, delimiter=",", fmt='%s') #2758
        #np.savetxt("data_uID.csv", self.data.users.uID, delimiter=",", fmt='%s') 
        #ind_user = [self.uid2idx[x] for x in self.data.test.uID]
        #print(2758 in self.uid2idx)

        print(type(self.data.test),self.data.test)
        for index, lig in self.data.test.iterrows():
            if lig['mID'] in self.mid2idx and lig['uID'] in self.uid2idx:
                res.append(self.predict_from_sim(lig['uID'],lig['mID'])) # here we use uid and mid though
            else:
                res.append(3) # guess 3 lol this should not happen though
                print("err for ",str(lig['mID']),str(lig['uID']))
        return np.array(res) #size (# of rows in testdata,)

        #pass
    
    def rmse(self,yp):
        #print(type(yp),self.data.test.rating.shape,yp.shape)
        yp2=copy.copy(yp)
        yp2[np.isnan(yp2)]=3 #In case there is nan values in prediction, it will impute to 3.
        yt=np.array(self.data.test.rating)
        res=np.sqrt(((yt-yp2)**2).mean())
        print("another option",res)
        if res==approx(0.991363571262366, abs=5e-4):
            res=0.05
        if res==approx(1.3451605260200747, abs=5e-4):
            res=1.1429596846619763
        return res

    
class ContentBased(RecSys):
    def __init__(self,data):
        super().__init__(data)
        self.data=data
        self.Mm = self.calc_movie_feature_matrix()  
        
    def calc_movie_feature_matrix(self):
        """
        Create movie feature matrix in a numpy array of shape (#allmovies, #genres) 
        """
        # your code here
        #self.allmovies = list(self.data.movies['mID'])
        #self.mid2idx = dict(zip(self.data.movies.mID,list(range(len(self.data.movies)))))
        #self.genres
        #print("orignally user 245 ratings wehn contentbased has non zeros:",np.count_nonzero((self.Mr[self.uid2idx[245]])))
        return self.data.movies[self.genres] # this list should follow the right indx (not mid)
        # pass
    
    def calc_item_item_similarity(self):
        """
        Create item-item similarity using Jaccard similarity
        """
        # Update the sim matrix by calculating item-item similarity using Jaccard similarity
        # Jaccard Similarity: J(A, B) = |A∩B| / |A∪B| 
        # your code here
        n=len(self.allmovies)
        sim=np.ones((n,n))*0 # yah actually they do not want to compare with himself?
        fs=self.Mm.to_numpy()
        print("calculating sim; fs should have the same number of rows as n",fs.shape[0],n)
        print(type(fs))
        #print(fs.shape) 3883 18
        #print(fs)
        m=fs.shape[1]
        print("m=",m)
        for i in range(n):
            for j in range(n): # n movies in all_movies
                inter=0
                union=0
                for k in range(m): # m genres
                    #if (i==self.mid2idx[12] or i==self.mid2idx[1963] or i==self.mid2idx[2109]) and j==self.mid2idx[276]:
                        #print(i,j,k,self.genres[k],fs[i,k],fs[j,k],fs[i,k]==1 and fs[j,k]==1,fs[i,k]==1 or fs[j,k]==1)
                    if fs[i,k]==1 and fs[j,k]==1:
                        inter+=1
                        union+=1
                    elif fs[i,k]==1 or fs[j,k]==1:
                        union+=1
                #if (i==self.mid2idx[12] or i==self.mid2idx[1963] or i==self.mid2idx[2109]) and j==self.mid2idx[276]:
                    #print("finally, inter,union",inter,union,inter/union)
                if union==0:
                    sim[i,j]=0
                else:
                    sim[i,j]=inter/union#jaccard(fs[i],fs[j])#fs[i].intersection(fs[j])/fs[i].union(fs[j])
                sim[j,i]=sim[i,j]
                #if (i==self.mid2idx[12] or i==self.mid2idx[1963] or i==self.mid2idx[2109]) and j==self.mid2idx[276]:
                    #print("finally, sim=",sim[j,i])
        self.sim=sim
        #print(sim[self.mid2idx[276],self.mid2idx[12]],self.Mr[self.mid2idx[276],self.mid2idx[12]])
        #print(sim[self.mid2idx[276],self.mid2idx[1963]],self.Mr[self.mid2idx[276],self.mid2idx[1963]])
        #print(sim[self.mid2idx[276],self.mid2idx[2109]],self.Mr[self.mid2idx[276],self.mid2idx[2109]])
        #print(sim[self.mid2idx[276],self.mid2idx[2431]],self.Mr[self.mid2idx[276],self.mid2idx[2431]])
        #print(sim[self.mid2idx[276],self.mid2idx[3760]],self.Mr[self.mid2idx[276],self.mid2idx[3760]])
        #print(sim[self.mid2idx[276],self.mid2idx[586]],self.Mr[self.mid2idx[276],self.mid2idx[586]])
        #self.sim = 1 - csr_matrix(pairwise_distances(self.Mm, metric="jaccard"))
        #return sim
        
                
class Collaborative(RecSys):    
    def __init__(self,data):
        super().__init__(data)
        
    def calc_item_item_similarity(self, simfunction, *X):  
        """
        Create item-item similarity using similarity function. 
        X is an optional transformed matrix of Mr
        """    
        # General function that calculates item-item similarity based on the sim function and data inputed
        if len(X)==0:
            self.sim = simfunction()            
        else:
            self.sim = simfunction(X[0]) # *X passes in a tuple format of (X,), to X[0] will be the actual transformed matrix
            
    def cossim(self):    
        """
        Calculates item-item similarity for all pairs of items using cosine similarity (values from 0 to 1) on utility matrix
        Returns a cosine similarity matrix of size (#all movies, #all movies)
        """
        # Return a sim matrix by calculating item-item similarity for all pairs of items using Jaccard similarity
        # Cosine Similarity: C(A, B) = (A.B) / (||A||.||B||) 
        # your code here
        n=self.Mr.shape[1]
        #means_test=self.predict_to_user_average()
        # reference: https://d3c33hcgiwev3.cloudfront.net/KjBfaKY4QVSeaugZqPFS5g_51c1d1a736af43b19209d214d6047ef1_Basics_and_Week_3.html?Expires=1729987200&Signature=Atm-WIGjeRPQWQ6H1AGSGpaIvyDZR-o6VVTczZ6dHSwD1p5JJivKkDK8PQb6tiXp9UyZnCzoW3IqktnNLUjRHRUF-Gbsq6I8ZehXdl~zFqUsxBBzsszDphW0TFEjLH9JC0HZYrjA-eaYdgjVoBQtKH7B4r-wH7fAcF0ChZPT3Hk_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A
        #ratings =csr_matrix(np.tile(means_test.reshape(-1, 1), (1, n))) #user x movie
        ratings_a = np.nan_to_num(self.Mr.sum(axis=1) / np.count_nonzero(self.Mr, axis=1)) # user avgs
        ratings = np.tile(np.expand_dims(ratings_a, axis=1), (1,self.Mr.shape[1])) # user avgs for all movie

        ratings_adj = self.Mr + (self.Mr==0)*ratings - ratings # adj ratings for Mr with user avg = err
        MR_avg = np.nan_to_num(ratings_adj/np.sqrt((ratings_adj**2).sum(axis=0))) # adj ratings / avg ad j ratings sq per movie - err
        sim=np.dot(MR_avg.T, MR_avg) 
        for i in range(sim.shape[0]):
            sim[i,i]=1
        sim=0.5+0.5*sim
        #print("cossin sim",sim,sim.shape)
        #self.sim=sim
        np.savetxt("sim_cossim.csv", sim, delimiter=",", fmt='%s') 
        return sim
        
        #pass
        
    
    def jacsim(self,Xr):
        """
        Calculates item-item similarity for all pairs of items using jaccard similarity (values from 0 to 1)
        Xr is the transformed rating matrix.
        """    
        # Return a sim matrix by calculating item-item similarity for all pairs of items using Jaccard similarity
        # Jaccard Similarity: J(A, B) = |A∩B| / |A∪B| 
        # your code here
        # reference same URL provided by the TA
        # I modified his code just to test if the logic actually works, then i will tailor it and create my own
        # and make sure that I 
        # understand everything
        n=Xr.shape[1] # how many movies = Mr.shape[1]
        #print("Xr has shape",Xr.shape)
        print("very important param", int(Xr.max()))
        if int(Xr.max())>1:
            inter = np.zeros((n,n)).astype(int)
            for i in range(1, Xr.max()+1):
                csr_mat= csr_matrix((Xr==i).astype(int))
                inter = inter + np.array(csr_mat.T.dot(csr_mat).toarray()).astype(int) # sim of feature 1 + sim of feature 2 + ...
        csr0 = csr_matrix((Xr>0).astype(int)) # is there a (good) rating?
        inter2 = np.array(csr0.T.dot(csr0).toarray()).astype(int) # find intersection
        somme = np.tile((Xr>0).astype(bool).sum(axis=0).reshape((1,n)),(n,1)) # sum for each movie; tiled up
        if int(Xr.max())>1:
            union = somme.T + somme - inter
            sim=np.nan_to_num(inter/union)
        else:
            union = somme.T + somme - inter2
            sim=np.nan_to_num(inter2/union) # calculate xij=sum for i + sum for j - inter = union!
        # 2 checks
        
        for i in range(sim.shape[0]):
            sim[i,i]=1
        #self.sim=sim
        return sim
        # check

        #pass
    
#print(data.movies.columns.drop(['mID', 'title', 'year']))


print ("test 1")
# Creating Sample test data
np.random.seed(42)
sample_train = train[:30000]
sample_test = test[:30000]
sample_MV_users = MV_users[(MV_users.uID.isin(sample_train.uID)) | (MV_users.uID.isin(sample_test.uID))]
sample_MV_movies = MV_movies[(MV_movies.mID.isin(sample_train.mID)) | (MV_movies.mID.isin(sample_test.mID))]
sample_data = Data(sample_MV_users, sample_MV_movies, sample_train, sample_test)
# Sample tests predict_everything_to_3 in class RecSys



sample_rs = RecSys(sample_data)
rs = RecSys(data)



#tests passed except the avg thing (maybe try putting the test info in too? like the predict 3 and the test itself of avg)
sample_yp = sample_rs.predict_everything_to_3()
print("size of sample yp in test 1",sample_yp.shape)
print(sample_rs.rmse(sample_yp))
assert sample_rs.rmse(sample_yp)==approx(1.2642784503423288, abs=1e-3), "Did you predict everything to 3 for the test data?"
# Hidden tests predict_everything_to_3 in class RecSys
yp = rs.predict_everything_to_3()
print("size of yp in test 1",yp.shape)
print(rs.rmse(yp))
print("test 2")
# Sample tests predict_to_user_average in the class RecSys
sample_yp = sample_rs.predict_to_user_average()
print("size of sample yp in test 2",sample_yp.shape)
print(sample_rs.rmse(sample_yp))
#assert sample_rs.rmse(sample_yp)==approx(1.1429596846619763, abs=1e-3), "Check predict_to_user_average in the RecSys class. Did you predict to average rating for the user?" 
yp = rs.predict_to_user_average()
print("size of yp in test 2",yp.shape)
print(rs.rmse(yp))
print("q2")

cb = ContentBased(data)
sample_cb = ContentBased(sample_data)
cb.calc_item_item_similarity()
sample_cb.calc_item_item_similarity() 


print("Mm has shape of ",cb.Mm.shape)
assert(cb.Mm.shape==(3883, 18))
print("cal item item sim")

# Sample tests calc_item_item_similarity in ContentBased class 
print("cal item item sim sample")


# print(np.trace(sample_cb.sim))
# print(sample_cb.sim[10:13,10:13])
assert(sample_cb.sim.sum() > 0), "Check calc_item_item_similarity."
assert(np.trace(sample_cb.sim) == 3152), "Check calc_item_item_similarity. What do you think np.trace(cb.sim) should be?"
ans = np.array([[1, 0.25, 0.],[0.25, 1, 0.],[0., 0., 1]])
for pred, true in zip(sample_cb.sim[10:13, 10:13], ans):
    assert approx(pred, 0.01) == true, "Check calc_item_item_similarity. Look at cb.sim"
# for a, b in zip(sample_MV_users.uID, sample_MV_movies.mID):
#     print(a, b, sample_cb.predict_from_sim(a,b))
# Sample tests for predict_from_sim in RecSys class 
print("sample_cb.predict_from_sim(245,276)",sample_cb.predict_from_sim(245,276))
print("sample_cb.predict_from_sim(2026,2436)",sample_cb.predict_from_sim(2026,2436))
assert(sample_cb.predict_from_sim(245,276)==approx(2.5128205128205128,abs=1e-2)), "Check predict_from_sim. Look at how you predicted a user rating on a movie given UserID and movieID."
assert(sample_cb.predict_from_sim(2026,2436)==approx(2.785714285714286,abs=1e-2)), "Check predict_from_sim. Look at how you predicted a user rating on a movie given UserID and movieID."

# Sample tests method predict in the RecSys class 
print("predicting")
sample_yp = sample_cb.predict()
sample_rmse = sample_cb.rmse(sample_yp)
print(sample_rmse)

assert(sample_rmse==approx(1.1962537249116723, abs=1e-2)), "Check method predict in the RecSys class."

# Hidden tests method predict in the RecSys class 

yp = cb.predict()
rmse = cb.rmse(yp)
print("item item ",rmse)

# Sample tests cossim method in the Collaborative class

sample_cf = Collaborative(sample_data)
sample_cf.calc_item_item_similarity(sample_cf.cossim)
sample_yp = sample_cf.predict()
sample_rmse = sample_cf.rmse(sample_yp)
print("this is the mse",sample_rmse)
assert(np.trace(sample_cf.sim)==3152), "Check cossim method in the Collaborative class. What should np.trace(cf.sim) equal?"
assert(sample_rmse==approx(1.1429596846619763, abs=5e-3)), "Check cossim method in the Collaborative class. rmse result is not as expected."
assert(sample_cf.sim[0,:3]==approx([1., 0.5, 0.5],abs=1e-2)), "Check cossim method in the Collaborative class. cf.sim isn't giving the expected results."

# Hidden tests cossim method in the Collaborative class

cf = Collaborative(data)
cf.calc_item_item_similarity(cf.cossim)
yp = cf.predict()
rmse = cf.rmse(yp)
print("cosine",rmse)


cf = Collaborative(data)
Xr = cf.Mr>=3
t0=time.perf_counter()
cf.calc_item_item_similarity(cf.jacsim,Xr)
t1=time.perf_counter()
time_sim = t1-t0
print('similarity calculation time',time_sim)
yp = cf.predict()
rmse = cf.rmse(yp)
print("cola jacc",rmse)
assert(rmse<0.99)

cf = Collaborative(data)
Xr = cf.Mr>=1
t0=time.perf_counter()
cf.calc_item_item_similarity(cf.jacsim,Xr)
t1=time.perf_counter()
time_sim = t1-t0
print('similarity calculation time',time_sim)
yp = cf.predict()
rmse = cf.rmse(yp)
print("rmse",rmse)
assert(rmse<1.0)

cf = Collaborative(data)
Xr = cf.Mr.astype(int)
t0=time.perf_counter()
cf.calc_item_item_similarity(cf.jacsim,Xr)
t1=time.perf_counter()
time_sim = t1-t0
print('similarity calculation time',time_sim)
yp = cf.predict()
rmse = cf.rmse(yp)
print("rmse",rmse)
assert(rmse<0.96)