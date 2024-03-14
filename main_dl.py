#csv input 
import pandas as pd 
data = pd.read_csv('koreansatscoreset.csv')
data = data.dropna()
ydata = data['admit'].values 
xdata = []
for i, rows in data.iterrows():
    xdata.append([rows['eascore'],rows['adms'],rows['rank']])

import tensorflow as tf
import numpy as np 
from tkinter import*
import tkinter
from ttkbootstrap.constants import*
import ttkbootstrap as ttk

#datainput 
def inputdata():
    inputeascore = float(entry1.get())
    inputadms = int(entry2.get())
    inputrate = int(entry3.get())
    
    #neural network
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dense(128, activation='tanh'),
        tf.keras.layers.Dense(256, activation='tanh'),
        tf.keras.layers.Dense(512, activation='tanh'),
        tf.keras.layers.Dense(1, activation='sigmoid'),    
    ])

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

    #learning    
    model.fit(np.array(xdata),np.array(ydata),epochs=10000)
    #output
    Prediction = model.predict( [ [inputeascore,inputadms,inputrate] ] ) 
    label["text"] = Prediction

#shutdown fun  
def shutdown():
    window.destroy()

#UI (tkinter)
window = tkinter.Tk()
window.title("수시 합격 예측 딥러닝 ")
window.geometry("500x400")
window.resizable(False,False)   
style = ttk.Style("darkly")

#text 
labelea=ttk.Label(window,text="평균 등급",font=('Arial',10))
labelea.place(x=150,y=50)

labelad=ttk.Label(window,text="교과/종합/논술",font=('Arial',10))
labelad.place(x=130,y=100)

labelrate=ttk.Label(window,text="대학 랭크",font=('Arial',10))
labelrate.place(x=150,y=150)

#inputeascore
entry1 = ttk.Entry(width=10)
entry1.place(x=230,y=50)
#inputadms 
entry2 = ttk.Entry(width=10)
entry2.place(x=230,y=100)
#inputrate 
entry3 = ttk.Entry(width=10)
entry3.place(x=230,y=150)

#button
button = ttk.Button(window, text = "딥러닝 시작",command=inputdata,bootstyle=SUCCESS)
button.place(x=220,y=200)

#buttonshutdown 
button2 = ttk.Button(window, text = "   종료   ",command = shutdown,bootstyle = DANGER)
button2.place(x=220,y=250)



label=ttk.Label(window, width=100,borderwidth=5,relief='solid',font=('Arial',40))
label.pack(side='bottom')
window.mainloop()





