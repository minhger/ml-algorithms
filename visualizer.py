# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 17:16:16 2018

@author: MNN
"""
from tkinter import *
import tkinter as tk
from tkinter.colorchooser import askcolor
import random
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree 
import os 



# =============================================================================
#1. change image_b1 to button_color1
#2. color_button = button_color
#3. choose_size_burron = slider_size
# color1 = COLOR_BLUE
# def select_color1(self):

# =============================================================================
# source for gui https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06
class Paint():

    DEFAULT_COLOR = 'black'
    COLOR1 = "#99d9ea"
    COLOR2 = "#b5e61d"
    COLOR3 = "#ff8080"
    COLOR4 = "#fda503"
    COLOR5 = "#b98bd1"
    point_coord = []
    
    

    def __init__(self):
        self.root = Tk()
        self.root.title("Visualizer in Machine Learning")
                    
        #Slider
        self.slider_size = Scale(self.root, label='Point size',from_=5, to=10, orient=HORIZONTAL,font="Arial 10") 
        self.slider_size.grid(row=0, rowspan=2, column=0, columnspan=10)

        self.slider_radius = Scale(self.root, label="Dispersion",from_=10, to=100, orient=HORIZONTAL,font="Arial 10")
        self.slider_radius.grid(row=0, rowspan=2, column=10,columnspan=10)
        
        self.slider_density = Scale(self.root, label="Density",from_=1, to=5, orient=HORIZONTAL,font="Arial 10")
        self.slider_density.grid(row=0, rowspan=2, column=20,columnspan=10)
        
        #Buttons color
        self.color_button = Button(self.root, text='Colour Palette', padx=57,command=self.choose_color,font="Arial 10")
        self.color_button.grid(row=0, column=30, columnspan=5)
               
        self.canvas = Canvas(self.root)#), width=20, height=20)
        self.button_color1= Button(self.root, bg=self.COLOR1,padx=10,pady=1,command=lambda: self.select_color(self.COLOR1))
        self.button_color1.grid(row=1,column=30)
        self.button_color2= Button(self.root, bg=self.COLOR2,padx=10,pady=1,command=lambda: self.select_color(self.COLOR2))
        self.button_color2.grid(row=1,column=31)
        self.button_color3= Button(self.root, bg=self.COLOR3,padx=10,pady=1,command=lambda: self.select_color(self.COLOR3)) 
        self.button_color3.grid(row=1,column=32)
        self.button_color4= Button(self.root, bg=self.COLOR4,padx=10,pady=1,command=lambda: self.select_color(self.COLOR4)) 
        self.button_color4.grid(row=1,column=33)
        self.button_color5= Button(self.root, bg=self.COLOR5,padx=10,pady=1,command=lambda: self.select_color(self.COLOR5)) 
        self.button_color5.grid(row=1,column=34)    

        #Testdaten
        self.testdata = IntVar()
        self.testdata.set(0) 
        self.knn_testdata = Checkbutton(self.root, text="Test data", variable=self.testdata, font="Arial 10")
        self.knn_testdata.grid(row=0,rowspan=2,column=35,columnspan=7,sticky="w") 
        
        #Buttons Save/Reset 
        self.csv_button = Button(self.root, text='Download',height=2,width=8,command=self.to_csv, font="Arial 9")
        self.csv_button.grid(row = 0, rowspan=2,column=42,columnspan=5) #pad innerhalb des grids, ipad, zellgröße  
        self.reset_button = Button(self.root, text='Clear', height=2,width=6, command=self.reset, font="Arial 9")
        self.reset_button.grid(row=0, rowspan=2,column=47,columnspan=3)
        
        
        ## Cluster Section
        self.canvas = Canvas(self.root, width=280, height=60) #, bg = '#F5A9A9')
        self.canvas.create_text(140,17,text="Select algorithm:", font="Arial 15 bold") #fill="darkblue",font="Times 20 italic bold",
        self.canvas.grid(row=2, column=50,columnspan=7,sticky=tk.W)
    
        #Kmeans
        self.canvas = Canvas(self.root, width=200, height=20) #, bg = '#afeeee')
        self.canvas.create_text(100,10,text="Number of clusters:", font="Arial 11")
        self.canvas.grid(row=3, column=50,columnspan=5)
        self.text_kmeans = Text(self.root, height=1, width=7)
        self.text_kmeans.insert(END, "")
        self.text_kmeans.grid(row=3, column=55,columnspan=1)      
        self.kmeans_button = Button(self.root, text='K-Means', font="Arial 11",command=self.kmeans,bg="#D3D3D3")#kmeans.import_and_cluster)
        self.kmeans_button.grid(row=4, column=50,columnspan=9,ipadx=98)
        
        #Empty block
        self.canvas = Canvas(self.root, width=200, height=30) #, bg = '#FFFF00')
        self.canvas.create_text(100,10,text="")
        self.canvas.grid(row=5, column=50,columnspan=5,sticky=tk.W)
        
        #Hierarchical Clustering
        self.lkge = IntVar()
        self.lkge.set(0)
        self.hc_single_radio = Radiobutton(self.root, text="single", variable=self.lkge, value=0, font="Arial 11") # mit padx koordinate festlegen
        self.hc_single_radio.grid(row=6, column=51,columnspan=5,sticky="w")
        self.hc_complete_radio = Radiobutton(self.root, text="complete", variable=self.lkge, value=1, font="Arial 11")
        self.hc_complete_radio.grid(row=7, column=51,columnspan=5,sticky="w")
        self.hc_average_radio = Radiobutton(self.root, text="average", variable=self.lkge, value=2, font="Arial 11")
        self.hc_average_radio.grid(row=6, column=54,columnspan=5,sticky="w")
        self.hc_centroid_radio = Radiobutton(self.root, text="centroid", variable=self.lkge, value=3, font="Arial 11")
        self.hc_centroid_radio.grid(row=7, column=54,columnspan=5,sticky="w")
        self.canvas = Canvas(self.root, width=200, height=20) #, bg = '#afeeee')
        self.canvas.create_text(100,10,text="Number of clusters:", font="Arial 11")
        self.canvas.grid(row=8, column=50,columnspan=5)
        
        self.text_hc = Text(self.root, height=1, width=7)
        self.text_hc.insert(END, "")
        self.text_hc.grid(row=8, column=55,columnspan=1)
        self.hier_cluster_button = Button(self.root, text="Hierachichal Clustering", font="Arial 11",command=self.hier_cluster,bg="#D3D3D3")
        self.hier_cluster_button.grid(row=9, column=50,columnspan=9,ipadx=52)
        
        #Empty block
        self.canvas = Canvas(self.root, width=200, height=30) #, bg = '#FFFF00')
        self.canvas.create_text(100,10,text="")
        self.canvas.grid(row=10, column=50,columnspan=5,sticky=tk.W)
        
        ## Classification Section
        
        #k neighbours      
        self.canvas = Canvas(self.root, width=200, height=20) #, bg = '#afeeee')
        self.canvas.create_text(100,10,text="Number of neighbours:", font="Arial 11")
        self.canvas.grid(row=11, column=50,columnspan=5)
        
        self.knn_text = Text(self.root, height=1, width=7)
        self.knn_text.insert(END, "")
        self.knn_text.grid(row=11, column=55,columnspan=1)

        self.knn_button = Button(self.root, font="Arial 11",text='K Nearest Neighbour', command=self.knn,bg="#D3D3D3")
        self.knn_button.grid(row=12, column=50,columnspan=9,ipadx=60)  
    
        #Empty block
        self.canvas = Canvas(self.root, width=200, height=30) #, bg = '#FFFF00')
        self.canvas.create_text(100,10,text="")
        self.canvas.grid(row=13, column=50,columnspan=5,sticky=tk.W)
    
        # decision tree
        self.dt_button = Button(self.root, font="Arial 11",text='Decision Tree', command=self.d_tree,bg="#D3D3D3")
        self.dt_button.grid(row=14, column=50,columnspan=9,ipadx=82)
        
        
        ##initializing data arrays
        self.point_coords = []
        self.colors = []
        self.sizes = []
        self.testdata_coords = []
        
        #create drawing area
        self.c = Canvas(self.root, bg='white', width=800, height=600)
        self.c.grid(row=2, rowspan=80,columnspan=50)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.size = self.slider_size.get()
        self.radius_width = self.slider_radius.get()
        self.color = self.DEFAULT_COLOR
        self.c.bind('<B1-Motion>', self.paint)
        self.slider_radius.set(50)
        self.slider_density.set(5)
        self.iteration = 1 #for density
        self.change = False
        self.to_be_controlled = False 
        self.canvas_error = Canvas(self.root, width=230, height=22) 
        
        # for testdata
        self.old_x_testdata = None
        self.old_y_testdata = None

    # draw circles / rectangles in chosen color
    def choose_color(self):
        self.color = askcolor(color=self.color)[1]
        
    def select_color(self,color):
        self.color = color
                
    def get_cluster_colors(self):
        cluster_colors = list(set(self.colors)) # the original color
        add_colors= ['blue','green', 'red', 'yellow', 'orange', 'black','grey','brown'] # extend to have more cluster colors
        cluster_colors.extend(add_colors)
        return cluster_colors

# =============================================================================
#        ALGORITHMS
#        
#        Clustering:
#            - k means
#            - hierarchichal Clustering
#        Classification:
#            - decision tree
#            - k nearest neighbour
#
# =============================================================================
    def kmeans(self):
        # check if text field (number of clusters) is filled and handle exception
        try:
            k = int(self.text_kmeans.get("1.0",END))
            if k:
                error = "fixed"
                self.handle_error(error)
        except Exception:
            error= "Please input number of clusters."
            self.handle_error(error)
        coord_array = self.prepare_array(self.point_coords)#prepare array for clustering
        # k-means algo
        kmeans = KMeans(n_clusters=k) # k cluster number
        kmeans.fit(coord_array)
        clusters = kmeans.predict(coord_array) #predicted colors
        plt.scatter(coord_array[:, 0], coord_array[:, 1], c=clusters, s=50, cmap='viridis') #cmap für farbpalette
        # draw new points in predicted colors
        cluster_colors = self.get_cluster_colors() # needed in case number of clusters is more than used colors
        for i in range(0,len(coord_array[:,0])):
            x = coord_array[i,0]
            y = coord_array[i,1]
            index= self.c.create_oval(x-self.sizes[i], y-self.sizes[i], x+self.sizes[i], 
                                      y+self.sizes[i],outline=self.colors[i],
                                      fill = cluster_colors[clusters[i]],width=self.sizes[-1]/10)  

    def hier_cluster(self):
        # check if text field (number of clusters) is filled and handle exception
        try:
            k = int(self.text_hc.get("1.0",END))
            if k:
                error = "fixed"
                self.handle_error(error)
        except Exception:
            error= "Please input number of clusters."
            self.handle_error(error)
        coord_array = self.prepare_array(self.point_coords) #prepare array for clustering
        # hierarchichal clustering algo
        lkge = ["single","complete","average","centroid"]
        Z = linkage(coord_array, lkge[self.lkge.get()])
        clusters = fcluster(Z, k, criterion='maxclust') 
        for x in range(0,len(clusters)):
            clusters[x] = clusters[x] - 1
        plt.scatter(coord_array[:,0], coord_array[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
        # draw clusters with predicted colors
        cluster_colors = self.get_cluster_colors()     
        for i in range(0,len(coord_array[:,0])):
            x = coord_array[i,0]
            y = coord_array[i,1]
            index= self.c.create_oval(x-self.sizes[i], y-self.sizes[i], 
                                      x+self.sizes[i], y+self.sizes[i],
                                      outline=self.colors[i], fill = cluster_colors[clusters[i]],
                                      width=self.sizes[-1]/10) #outline=self.colors[i]

    def knn(self):
        # check if text field (number of neighbours) is filled and handle exception
        try:
            k = int(self.knn_text.get("1.0",END))
            if k:
                error = "fixed"
                self.handle_error(error)
        except Exception:
            error= "Please input number of neighbours."
            self.handle_error(error)
        #initialize features (x) and labels (y)
        y = self.colors
        X = np.array(self.point_coords)        
        # check if there are test data or only training data to predict
        if self.testdata.get() == 0:
            data = self.point_coords
        else:
            data = self.testdata_coords
        # k nearest neighbour algo
        neigh = KNeighborsClassifier(n_neighbors=k) ### noch umändern
        neigh.fit(X, y) 
        KNeighborsClassifier(...)
        cluster_colors= neigh.predict(data)             
        coord_array = self.prepare_array(data) # prepare array to draw new points
        # draw training data in predicted colors ###################################### leave out?
        if self.testdata.get() == 0:
            
            for i in range(0,len(coord_array[:,0])):
                x = coord_array[i,0]
                y = coord_array[i,1]
                index= self.c.create_oval(x-self.sizes[i], y-self.sizes[i], 
                                          x+self.sizes[i], y+self.sizes[i], 
                                          fill= cluster_colors[i],outline = self.colors[i],
                                          width=self.sizes[-1]/10) 
        # draw test data in predicted colors
        # shape is different
        else:
            for i in range(0,len(coord_array[:,0])):
                x = coord_array[i,0]
                y = coord_array[i,1]
                x1= x
                y1= y-(self.sizes[-1]*1.3)
                x2= x+(self.sizes[-1]*1.3)
                y2= y
                x3= x
                y3= y+(self.sizes[-1]*1.3)
                x4= x-(self.sizes[-1]*1.3)
                y4= y
                index= self.c.create_polygon(x1,y1,x2,y2,x3,y3,x4,y4,outline="black",
                                             fill = cluster_colors[i],
                                             width=self.sizes[-1]/3)
    
    def d_tree(self):
        #initialize features (x) and labels (y)                                      
        x = np.array(self.point_coords)                                                             
        y = self.colors
        # check if there are test data or only training data to predict 
        if self.testdata.get() == 0:
            data = self.point_coords
        else:
            data = self.testdata_coords
        # decision tree algo
        clf = tree.DecisionTreeClassifier()                                                                         
        clf = clf.fit(x, y)  
        cluster_colors = clf.predict(data)
        coord_array = self.prepare_array(data) # prepare array to draw predicted labels (colors)
        # draw training data in predicted colors
        if self.testdata.get() == 0:
            for i in range(0,len(coord_array[:,0])):
                x = coord_array[i,0]
                y = coord_array[i,1]
                index= self.c.create_oval(x-self.sizes[i], y-self.sizes[i], 
                                          x+self.sizes[i], y+self.sizes[i], 
                                          fill = cluster_colors[i], 
                                          outline = self.colors[i],width=self.sizes[-1]/10) 
        # draw test data in predicted colors 
        # shape is different
        else:
            for i in range(0,len(coord_array[:,0])):
                x = coord_array[i,0]
                y = coord_array[i,1]
                x1= x
                y1= y-(self.sizes[-1]*1.3)
                x2= x+(self.sizes[-1]*1.3)
                y2= y
                x3= x
                y3= y+(self.sizes[-1]*1.3)
                x4= x-(self.sizes[-1]*1.3)
                y4= y
                index= self.c.create_polygon(x1,y1,x2,y2,x3,y3,x4,y4,outline="black",fill = cluster_colors[i],
                                             width=self.sizes[-1]/3)   
                
                
                
                
# =============================================================================
# HELPERS   
# =============================================================================

    #
    def prepare_array(self, data):
        coord_array = np.zeros((len(data),2)) 
        j = 0
        
        for i in range(0,len(data)):
            coord_array[i][j] = data[i][j]
            coord_array[i][j+1] = data[i][j+1]
        return coord_array
            
    #missing inputs for algorithms
    def handle_error(self,error):
        if error == "fixed":
            self.canvas_error.delete("all")
        else:
            self.canvas_error.create_text(100,10,text=error, font="Arial 8 bold", fill="red")
            self.canvas_error.grid(row=75, column=50,columnspan=15,sticky=tk.E)   
        
    #draw circles / rectangles
    def paint(self, event):
        self.radius_width = self.slider_radius.get() #get selected radius
        # monitor changes in settings so that saved values for self.old_x and self.old_y are skipped 
        if self.to_be_controlled == True:
            if len(self.colors) > 0:        
                if self.colors[-1] != self.color or self.sizes[-1] != int(self.slider_size.get()):
                    self.change = True
                else:
                    self.change = False
        # skip iterations for different densities
        self.density = self.slider_density.get() * -1 + 6 # reversing order
        self.iteration = self.iteration + 1
        # if no changes, points can be drawn
        if self.change == False: 
            self.to_be_controlled = True 
            # density
            if self.iteration % self.density == 0: 
                if self.old_x and self.old_y:
                    self.sizes.append(int(self.slider_size.get())) 
                    x= self.old_x+random.randint(-self.radius_width, self.radius_width)
                    y= self.old_y+random.randint(-self.radius_width, self.radius_width) 
                    # draw circles if checkbox testdata is not selected
                    if self.testdata.get() == 0:
                        index= self.c.create_oval(x-self.sizes[-1], y-self.sizes[-1], x+self.sizes[-1], 
                                                  y+self.sizes[-1], outline = self.color,width=self.sizes[-1]/3)
                        self.point_coords.append([x,y])
                        self.colors.append(self.color)
                    # draw rectangles if testdata is selected
                    else:
                        x1= x
                        y1= y-(self.sizes[-1]*1.3)
                        x2= x+(self.sizes[-1]*1.3)
                        y2= y
                        x3= x
                        y3= y+(self.sizes[-1]*1.3)
                        x4= x-(self.sizes[-1]*1.3)
                        y4= y
                        index= self.c.create_polygon(x1,y1,x2,y2,x3,y3,x4,y4, outline="black", fill = "",
                                                     width=self.sizes[-1]/3)
                        self.testdata_coords.append([x,y])
        else:
            self.change = False
            self.to_be_controlled = False
            self.old_x = event.x
            self.old_y = event.y
        self.old_x = event.x
        self.old_y = event.y
        
        
    # download coordinates as csv file
    def to_csv(self):
        df = pd.DataFrame()
        df = df.append(pd.DataFrame({
                'point': self.point_coords,
                'color': self.colors,
                }))
        location = self.get_download_path()
        location = location.replace('\\', '/')
        #location = './'#C:/Users/MNN/Desktop/minh uni/6. SM/ba/' 
        """location = os.getcwd() + '\\'
        location = location.replace('\\', '/')"""
        print('File created in: ' + location)
        csv_name = 'data' + '.csv'
        print(csv_name)
        df.to_csv(location+"/"+csv_name, encoding='utf-8', sep=';', index=True)

    # clearing drawing space
    def reset(self): 
        self.old_x, self.old_y = None, None
        self.c.delete('all')
        self.canvas_error.delete("all")
        self.point_coords = []
        self.sizes = []
        self.colors = []
        self.testdata_coords = []
        self.iteration = 0 
        self.to_be_controlled, self.change = None, None
     
        
    # get download path
    # https://stackoverflow.com/questions/35851281/python-finding-the-users-downloads-folder
    def get_download_path(self):
        if os.name == 'nt':
            import winreg
            sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
            downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
                location = winreg.QueryValueEx(key, downloads_guid)[0]
            return location
        else:
            return os.path.join(os.path.expanduser('~'), 'downloads')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    Paint()
