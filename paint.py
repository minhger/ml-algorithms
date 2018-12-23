from tkinter import *
from tkinter.colorchooser import askcolor
import random
import pandas as pd
#import kmeans
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class Paint(object):

    DEFAULT_COLOR = 'black'
    point_coord = []

    def __init__(self):
        self.root = Tk()
        self.root.title("Airbrush")
        
        #create widgets
        self.choose_size_button = Scale(self.root, label='Point size',from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=0)
        
        self.choose_radius_button = Scale(self.root, label='Variance', from_=1, to=20, orient=HORIZONTAL)
        self.choose_radius_button.grid(row=0, column=1)
        
        self.color_button = Button(self.root, text='Colour', command=self.choose_color)
        self.color_button.grid(row=0, column=2)
        
        self.csv_button = Button(self.root, text='Download data as CSV', command=self.to_csv)
        self.csv_button.grid(row = 0, column = 3)
        
        self.reset_button = Button(self.root, text='Clear', command=self.reset)
        self.reset_button.grid(row=0, column=4)
        
        self.canvas = Canvas(self.root, width=160, height=20) #, bg = '#afeeee')
        self.canvas.create_text(100,10,text="Clustering Algorithms") #fill="darkblue",font="Times 20 italic bold",
        self.canvas.grid(row=1, column=0)
        self.canvas = Canvas(self.root, width=150, height=20) #, bg = '#afeeee')
        self.canvas.create_text(100,10,text="Number of clusters")
        self.canvas.grid(row=2, column=0)
        
        self.cluster_text = Text(self.root, height=1, width=10)
        self.cluster_text.insert(END, "")
        self.cluster_text.grid(row=2, column=1)
        
        
        self.kmeans_button = Button(self.root, text='K-Means', command=self.kmeans)#kmeans.import_and_cluster)
        self.kmeans_button.grid(row=2, column=2)
        
        self.hier_cluster_button = Button(self.root, text="Hierachichal Clustering", command=self.hier_cluster)
        self.hier_cluster_button.grid(row=3, column=2)
        
        #initializing data arrays
        self.point_coords = []
        self.colors = []
        self.sizes = []

        
        #create canvas
        self.c = Canvas(self.root, bg='white', width=600, height=600)
        self.c.grid(row=4, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.size = self.choose_size_button.get()
        self.radius_width = self.choose_radius_button.get()
        self.color = self.DEFAULT_COLOR
        self.c.bind('<B1-Motion>', self.paint)

    def choose_color(self):
        self.color = askcolor(color=self.color)[1]
    
    def kmeans(self):
        clusters = int(self.cluster_text.get("1.0",END))
        #clusters = 2
        coord_array = np.zeros((len(self.point_coords),2)) 
        j = 0
        for i in range(0,len(self.point_coords)):
            coord_array[i][j] = self.point_coords[i][j]
            coord_array[i][j+1] = self.point_coords[i][j+1]
            
        kmeans = KMeans(n_clusters=clusters) # K Clusterzahl
        kmeans.fit(coord_array)
        y_kmeans = kmeans.predict(coord_array)
        plt.scatter(coord_array[:, 0], coord_array[:, 1], c=y_kmeans, s=50, cmap='viridis') #cmap für farbpalette
        
        cluster_colors = ['blue','green', 'red', 'yellow', 'orange']       
        print(len(coord_array[:,0]))
        print(len(self.sizes))
        print(self.sizes)
        for i in range(0,len(coord_array[:,0])):
            x = coord_array[i,0]
            y = coord_array[i,1]
            cluster = y_kmeans[i]
            index= self.c.create_oval(x-self.sizes[i], y-self.sizes[i], x+self.sizes[i], y+self.sizes[i], fill = cluster_colors[cluster]) # ist der index

    def hier_cluster(self):
        clusters = int(self.cluster_text.get("1.0",END))
        coord_array = np.zeros((len(self.point_coords),2)) 
        j = 0
        for i in range(0,len(self.point_coords)):
            coord_array[i][j] = self.point_coords[i][j]
            coord_array[i][j+1] = self.point_coords[i][j+1]

        cluster = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward')  
        cluster.fit_predict(coord_array)  
        
        clusters = cluster.labels_

        plt.scatter(coord_array[:,0],coord_array[:,1], c=cluster.labels_, cmap='rainbow') 
        cluster_colors = ['blue','green', 'red', 'yellow', 'orange']       
        print(len(coord_array[:,0]))
        print(len(self.sizes))
        print(self.sizes)
        for i in range(0,len(coord_array[:,0])):
            x = coord_array[i,0]
            y = coord_array[i,1]
            cluster = clusters[i]
            index= self.c.create_oval(x-self.sizes[i], y-self.sizes[i], x+self.sizes[i], y+self.sizes[i], fill = cluster_colors[cluster]) # ist der index

        
    def paint(self, event):
#        self.sizes.append(int(self.choose_size_button.get()))
        #print(self.sizes)
        self.radius_width = self.choose_radius_button.get()
        #paint_color = self.color 
        #generate random numbers
        #radius = self.radius_width
        #size = self.line_width
        if self.old_x and self.old_y:
            self.sizes.append(int(self.choose_size_button.get()))
            x= self.old_x+random.randint(-self.radius_width, self.radius_width)
            y= self.old_y+random.randint(-self.radius_width, self.radius_width)
            #print("x= "+str(x))
            #print("y= "+str(y))         
            index= self.c.create_oval(x-self.sizes[-1], y-self.sizes[-1], x+self.sizes[-1], y+self.sizes[-1], outline = self.color) # ist der index
            color = self.c.itemcget(index, "outline") #get color
            point_coord = self.c.coords(index) #4 koordinatenpunkte des kreises

            #i = self.c.create_oval(x-size, y-size, x+size, y+size, outline = "red")
            #################### http://effbot.org/tkinterbook/canvas.htm#Tkinter.Canvas.create_oval-method
            self.point_coords.append([x,y])#point_coord[:2])
            self.colors.append(color)
            
        
            
        self.old_x = event.x
        self.old_y = event.y
    
    def to_csv(self):
        #Dataframe
        df = pd.DataFrame()
        df = df.append(pd.DataFrame({
                'point': self.point_coords,
                'color': self.colors,
                }))
        location = 'C:/Users/MNN/Desktop/minh uni/6. SM/ba/' #'Desktop/minh uni/6. SM/ba/'
        csv_name = 'data' + '.csv'
        df.to_csv(location+csv_name, encoding='utf-8', sep=';', index=True)
        print("here")

    def reset(self): 
        self.old_x, self.old_y = None, None
        self.c.delete('all')
        self.point_coords = []
        self.sizes = []
        self.colors = []
        
        

if __name__ == '__main__':
    Paint()
