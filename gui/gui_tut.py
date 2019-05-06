from tkinter import *

root = Tk()

# # tut 5
# label_1 = Label(root, text="name")
# label_2 = Label(root, text="pass")

# entry_1 = Entry(root)
# entry_2 = Entry(root)

# label_1.grid(row=0,sticky=E)
# label_2.grid(row=1,sticky=E)

# entry_1.grid(row=0,column=1)
# entry_2.grid(row=1,column=1)

# c = Checkbutton(root,text="Keep me login")
# c.grid(columnspan=2)

# # tut 6
def printHello():
    print("hello world")

def printName(event):
    print("hello nguyen")

while True:
    button_1 = Button(root, text="Print Hello", command=printHello)
    button_1.pack()
    button_2 = Button(root, text="Print Name")
    button_2.bind("<Button-1>",printName)
    button_2.pack()

# tut 7

# class Buttons:
#     def __init__(self,master):
#         frame = Frame(master)
#         frame.pack()

#         self.printButton = Button(frame, text="Print",command=self.printMessage)
#         self.printButton.pack(side=LEFT)

#         self.quitButton = Button(frame, text="Quit",command=frame.quit)
#         self.quitButton.pack(side=LEFT)

#     def printMessage(self):
#         print("hello")

# b = Buttons(root)

# status =Label(root,text="Do nothing....",bd=1,relief=SUNKEN,anchor=W)
# status.pack(side=BOTTOM, fill=X)
# root.mainloop()
