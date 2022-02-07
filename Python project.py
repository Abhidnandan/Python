
#IMPORT NUMPY FOR MATRIX OPERATION
import numpy as np
#DEFINING CLASS MATRIX
class matrix:

    def load_from_csv(self, filename : str):
        
        #Loads the matrix from a CSV and reads the matrix line by line
        file = open(filename, 'r')
        lin = file.readlines()

        data = []
        for j in lin:
            var = list(map(float, j.split(',')))
            data.append(var)

        self.array_2d = np.array(data)

        file.close()

        return self.array_2d

    
    
    def __init__(self, filename=None, arr=None):
        
       # Constructor function allows loading with matrix(<filename>)

        #if array is given the array is loaded in to array_2b or if file is given it wil loaded in to array_2d
        
        if filename is not None:
            self.array_2d = self.load_from_csv(filename)

        if arr is not None:
            self.array_2d = arr


    def standardise(self):

        #it standardizes the array using equation""
        avg = np.mean(self.array_2d, axis =0)
        maxVal = np.max(self.array_2d, axis =0)
        minVal = np.min(self.array_2d, axis =0)
        '''
        print('Average  :',average.shape,'\n',average,'\n')
        print('Max  :',average.shape,'\n',maxValue,'\n')
        print('Min  :',average.shape,'\n', minValue,'\n')
        '''
        standardizedArray = []
        
        for eachrow in self.array_2d:
            stdRow = []
            stdStr = ''
            for colNo in range(len(eachrow)):
                stdRow.append((eachrow[colNo] - avg[colNo]) / (maxVal[colNo] - minVal[colNo]))
            standardizedArray.append(stdRow)
        return np.array(standardizedArray)

    def get_distance(self, other_matrix, weights, beta):
        """
        it gets the distance of an array and calculate the distances
        between thz row and all the rows in other matrix"""

        if self.array_2d.shape[0] == 1:
            distance = np.zeros((other_matrix.array_2d.shape[0], 1))

            # for all rows in other_matrix
            for k in range(other_matrix.array_2d.shape[0]):
                c1 = self.array_2d.reshape(1, -1)
                c2 = other_matrix.array_2d[k, :].reshape(1, -1)
                difve = np.square(c1 - c2)
                powerwts = np.power(weights, beta).reshape(-1, 1)
                dists = np.sum(np.dot(powerwts, difve))
                distance[k, :] = dists

            return distance

    def get_count_frequency(self):
        """
       And here we Counts the frequency of elements in the matrix
        """
        if self.array_2d.shape[1] == 1:
            dictionarymaps = {}
            for j in self.array_2d:
                if j[0] in dictionarymaps:
                    dictionarymaps[j[0]] += 1
                else:
                    dictionarymaps[j[0]] = 0
            
            return dictionarymaps

def get_initial_weights(m):
    """it gets the initial weight and generate random array
    and divides by the total sum"""
    wts = np.random.random(m)
    wts /= np.sum(wts)
    
    return wts


def get_centroids(mat : matrix, S : np.ndarray, K : int):
    """
    Here we will updates the centroids and finds the nearby points
    to find the new centroid mean.
    """
    mat.standardise()

    cent = np.zeros((K, mat.array_2d.shape[1]))
    for i in range(K):
        corr_rows = []
        for k in range(S.shape[0]):
            if(S[k, 0] == i):
                corr_rows.append(mat.array_2d[k, :])

        cent[i, :] = sum(corr_rows)/len(corr_rows)


    return matrix(arr=cent)



def get_groups(mat, K, beta):
    """
    Makes groups with data matrix
    """
    mat.standardise()
    n, m = mat.array_2d.shape
    weights = get_initial_weights(m)
    centroids = matrix()

    # Create a matrix S with n rows and 1 column, 
    S = np.zeros((n, 1))

    # selecting K different rows at random
    rows = np.random.choice(n, K, replace=False)

    # populating centroids matrix with K different selected rows
    centroids.array_2d = mat.array_2d[rows, :].copy()


    while True:

        # here we will keep track if centroids change
        changes = 0

        # for every data point
        for k in range(n):
            rmat = matrix(arr = mat.array_2d[k, :].reshape(1, -1))

            distance = rmat.get_distance(centroids, weights, beta)
            centroid = np.argmin(distance)

            if S[k, :] != centroid:
                changes += 1
                S[k, :] = centroid
            
        if changes == 0:
            return matrix(arr = S)
        
        
        # updating centroids and weights
        centroids = get_centroids(mat, S, K)
        weights = get_new_weights(mat, centroids, S, beta)





def get_new_weights(mat, centroids, S, beta):

    """
    Updates the existing weights by calculating dispersion.
    """
    # matrix initializing
    n, m = mat.array_2d.shape
    weights = np.zeros((1, m))
    K, _ = centroids.array_2d.shape


    # dispersion calculation
    dispersion = np.zeros((1, m))
    for h in range(K):
        for j in range(n):
            if(S[j, :] == h):
                dispersion += np.square(mat.array_2d[j, :] - centroids.array_2d[h, :])
    

    for k in range(m):
        if dispersion[0, k] == 0:
            weights[0, k] = 0
        else:
            values = 0
            for i in range(m):
                values += pow(dispersion[0, k]/dispersion[0, i], 1/(beta-1))
            weights[0, k] = values

    return weights


def run_test():
    m = matrix('Data.csv')
    for k in range(2,5):
        for beta in range(11,25):
            S = get_groups(m, k, beta/10)
            print(str(k)+'-'+str(beta)+'='+str(S.get_count_frequency()))
            

run_test()

