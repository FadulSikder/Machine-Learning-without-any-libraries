
import numpy as np
import pandas as pd
import math, sys
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("data_2c.csv",skiprows=1, header = None)
    data = df.to_numpy()
    X = data[:,:-1]
    Y = data[:,-1:]
    return X, Y

class Hierarchical_Clustering(object):
    def __init__(self,data,label, cluster_algo):
        self.data = data
        self.label = label
        self.cluster_algo = cluster_algo
        
        
        for i in [2,4,6,8]:
            print('\033[1m'+"Clustering algorihtm : ", cluster_algo,' with ', i,' cluster'+ '\033[0m')
            #print('\033[1m' + 'Number of cluster : ', i, ' '+ '\033[0m')
            self.number_of_clusters = i
            # Clustering
            self.cluster()
    
    def cluster(self):
        
        # Distance Matrix
        distance_matrix = self.get_distance_matrix(self.data)
        np.fill_diagonal(distance_matrix, sys.maxsize)



        #Finding the clusters
        array_clusters = self.divide_in_clusters(distance_matrix)
        
        # Getting n clusters and save them backward
        n = distance_matrix.shape[0] - self.number_of_clusters
        cluster = array_clusters[n]
        
        # Getting individual cluster
        unique_arr = np.unique(cluster)
        n_clusters = []
        for i in np.nditer(unique_arr):
            n_clusters.append(np.where(cluster == i))

        for j in range(len(n_clusters)):
            print("Cluster ", j + 1, " : ", n_clusters[j][0])
        
        # Plot clusters
        self.plot_cluster(n_clusters)
        # Plot Dendogram
        
        # Calulate error
        self.calculate_error(n_clusters)
        print("\n\n")
        
    def divide_in_clusters(self, distance_matrix):
        clusters = {}
        cluster_id = []
        row_id = 0
        col_id = 0
        
        for n in range(distance_matrix.shape[0]):
            cluster_id.append(n)
        
        clusters[0] = cluster_id.copy()
    
        # Min from the distance matrix
        for k in range(1, distance_matrix.shape[0]):
            min_val = sys.maxsize
            
            for i in range(distance_matrix.shape[0]):
                for j in range(distance_matrix.shape[1]):
                    if(distance_matrix[i][j] <= min_val):
                        min_val = distance_matrix[i][j]
                        row_id = i
                        col_id = j
            
            # Update the distance matrix
            for i in range(distance_matrix.shape[0]):
                if(i != col_id):
                    if(self.cluster_algo == "Average Linkage"):
                        temp = 0.5*(distance_matrix[col_id][i]+distance_matrix[row_id][i])
                    elif(self.cluster_algo == "Single Linkage"):
                        temp = min(distance_matrix[col_id][i],distance_matrix[row_id][i])
                    else:
                        temp = max(distance_matrix[col_id][i],distance_matrix[row_id][i])
                    
                    # Symmetric update of distance matrix
                    distance_matrix[col_id][i] = temp
                    distance_matrix[i][col_id] = temp
            
            for i in range (distance_matrix.shape[0]):
                distance_matrix[row_id][i] = sys.maxsize
                distance_matrix[i][row_id] = sys.maxsize
            
            minimum = min(row_id,col_id)
            maximum = max(row_id,col_id)

            for n in range(len(cluster_id)):
                if(cluster_id[n] == maximum):
                    cluster_id[n] = minimum

            clusters[k] = cluster_id.copy()

        return clusters
    
    # Distance matrix
    def get_distance_matrix(self, data):
        distance_matrix = np.zeros((data.shape[0],data.shape[0]))
        for i in range(distance_matrix.shape[0]):
            for j in range(distance_matrix.shape[0]):
                distance_matrix[i][j] = self.get_euclidean_distance(data[i], data[j])
        return distance_matrix
    
    # Euclidean distance
    def get_euclidean_distance(self, x1, x2):
        d = 0.0
        for i in range(0, len(x1)):
            d = d + (x1[i] - x2[i]) ** 2
        #print(d)
        return math.sqrt(d)
        
    def plot_cluster(self, n_clusters):
        plt3d = plt.figure(figsize=(16, 12), dpi=80).gca(projection='3d')
        p=0
        # Color for scatter plot blobs
        color = ['r','g','b','y','c','m','k','w']
        for i in range(len(n_clusters)):
            for j in np.nditer(n_clusters[i]):
                   plt3d.scatter(self.data[j,0], self.data[j,1],self.data[j,2], marker='o', c = color[p], s=50)
            p = p + 1
        plt3d.set_xlabel('Height')
        plt3d.set_ylabel('Weight')
        plt3d.set_zlabel('Age')
        plt.title('Hierarchical Clustering: '+self.cluster_algo+' for '+str(self.number_of_clusters)+' Cluster', fontsize=20, color='blue', loc='left')
        plt.show()

    def calculate_error(self, n_clusters):

        print('\033[1m'+"\nError Calculation"+ '\033[0m')
        total_correct = 0
        accuracy = 0
        overall_accuracy = 0
        for cls in range(0, len(n_clusters)):
            sub_cluster = n_clusters[cls][0]
            
            
            # Count Men\Women for each sub-cluster
            men = 0
            women = 0
            for i in range(len(sub_cluster)):
                id = sub_cluster[i]
                if(self.label[id] == 'M'):
                    men = men + 1
                else:
                    women = women + 1

            if(men > women):
                frequent_label = 'Men'
                frequent = men
            else:
                frequent_label = 'Women'
                frequent = women
            
            print("Cluster ", cls + 1, " : Men = ", men, ", Women = ", women,'. So, frequent label is',frequent_label)
            total_correct = int(frequent)
            accuracy = float(total_correct / len(sub_cluster))
            print("Accuracy for Cluster ", cls + 1, " : ", accuracy * 100, "%\n")
            overall_accuracy = overall_accuracy + accuracy*(len(sub_cluster)/len(self.label))
        
        print('\033[1m'+'Overall Accuracy for '+self.cluster_algo+' with '+str(self.number_of_clusters)+' Cluster Solution: ', overall_accuracy*100,' '+ '\033[0m')

def main():
    data,label = load_data()
    Hierarchical_Clustering(data,label, "Average Linkage")
    Hierarchical_Clustering(data,label, "Single Linkage")
    Hierarchical_Clustering(data,label, "Complete Linkage")
    return

if __name__ == '__main__':
    main()