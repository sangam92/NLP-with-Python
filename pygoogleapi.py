# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 22:48:51 2016

@author: sangam
"""
#import tkinter
import enchant
#from enchant.tokenize import get_tokenizer
#top=tkinter.Tk()
d=enchant.Dict("en_US")
f=enchant.Dict("fr_FR")
import speech_recognition as sr
#socgen_word= {XONE: }
#tknzr = get_tokenizer("en_US")
#spell = []
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Hi This is Kiku!!Say Something!!!")
    audio = r.listen(source)
   
    
    


try:
   
    print("You said: " + r.recognize_google(audio))
except sr.UnknownValueError:
     print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:

    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
#    top.mainloop()
    
b= r.recognize_google(audio)

print(b)

for word in b.split():
    a=d.check(word)
    #print(word)
    #print(a)
    
    if (a == True):
        print('english')
    else:
        for word in b.split():
            a=f.check(word)
        print('French')