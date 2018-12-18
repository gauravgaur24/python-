from tkinter import *
import mysql.connector

t=Tk()
t.geometry("500x600")
l1=Label(text="enter the name")
#l2=Label(text="enetr contact no  ")
l1.pack()
#l2.pack()

t1=Entry()
t1.pack()
l2=Label()
l2.pack()
def getdata():
    d=t1.get()
    print('you have entered :',d)
    l2.configure(text=d)
    con = mysql.connector.connect(host='localhost',user='root',password='root',database='dbhrms')
    cur =con.cursor()

    cur.execute("insert into tbl_user(name) values('"+d+"')")

    con.commit()
    con.close()
     

b1=Button(text='click me ',command=getdata)
b1.pack()
t.mainloop
