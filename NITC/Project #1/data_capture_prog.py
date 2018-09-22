import numpy as np
from PIL import ImageGrab
import keyboard,mouse,time
import pandas as pd


'''Edit these variable accordingly'''
start_x = 0
start_y = 110
end_x = 800
end_y = 550

##Listening to these keys only! and saved in this order
KEYS = ['w','s',75,77,'q','e']


dataset = pd.DataFrame(columns=['filename','keys_pressed','mouse_data'])
last = time.time()
j = 0 #Frame count


for i in range(10):
            time.sleep(1)
            print("Starting Frame Capture in ",10-i)

while(True):
    k_pressed = []
    screen = ImageGrab.grab(bbox=(start_x,start_y,end_x,end_y))
    screen_np = np.array(screen)
    filename='data'+str(time.time())+ '.jpg'
    screen.save(filename)
    for i in KEYS:
        k_pressed.append(keyboard.is_pressed(i))
    mice = [mouse.get_position()[0],mouse.get_position()[1],
            mouse.is_pressed(button='left'),mouse.is_pressed(button='middle'),mouse.is_pressed(button='right')]
    dataset.loc[j] = [[filename],k_pressed,mice]
    j = j +1

    '''Press esc to pause the capture for 10 secs and
       previous data in memory will recorded to disk also.'''
    if keyboard.is_pressed('Esc'):
        dataset.to_csv('dataset.csv',sep=',')
        print(j," Frames Done")
        for i in range(10):
            time.sleep(1)
            print("Continuing Frame Capture in ",10-i)
    print(j,'---loop took {} seconds'.format(time.time() - last))
    last = time.time()


#Just including these scancodes for other values if there is any prob with the KEYS   
s_codes = {'a':30,
     'b':48,
     'c':46,
     'd':32,
     'e':18,
     'f':33,
     'g':34,
     'h':35,
     'i':23,
     'j':36,
     'k':37,
     'l':38,
     'm':50,
     'n':49,
     'o':24,
     'p':25,
     'q':16,
     'r':19,
     's':31,
     't':20,
     'u':22,
     'v':47,
     'w':17,
     'x':45,
     'y':21,
     'z':44,
     'k_left':75,
     'k_right':77,
     'k_return':28,
     'k_up':72,
     'k_down':80,
     'n_1':79,
     'n_2':80,
     'n_3':81,
     'n_4':75,
     'n_5':76,
     'n_6':77,
     'n_7':71,
     'n_8':72,
     'n_9':73,
     'n_0':82,
     't_1':2,
     't_2':3,
     't_3':4,
     't_4':5,
     't_5':6,
     't_6':7,
     't_7':8,
     't_8':9,
     't_9':10,
     't_0':11,
     'l_shift':42,
     'r_shift':54,
     'tab': 15,
     'lr_ctrl':29, # note left and right shift has the same scancode value.
     'l_alt':56,
     'r_alt':541,
     'w_key':91
     
     }
