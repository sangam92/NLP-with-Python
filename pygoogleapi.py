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
import speech_recognition as sr
#tknzr = get_tokenizer("en_US")
#spell = []
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Hi This is Kiku!!Say Something!!!")
    audio = r.listen(source)
#    spell = audio
    
 #   for w in tknzr(spell):
  #      d.check(spell)

#    d.check(audio)

try:
   
    print("You said: " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:

    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    
#    top.mainloop()