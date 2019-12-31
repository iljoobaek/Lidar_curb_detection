import cv2
import numpy as np
key_map = {
    'interrupt_key': ord('i'),
    'continue_key': ord('c'),
    'next_step_key':ord('n'),
    'save_key':ord('s')
}
class VidHotKey():
    def __init__(self, saver = None, img = None):
        self.state = 'idle'
        self.saver = saver
        self.img = img

    def vid_hot_key(self,key):
        """ hot key setting for opencv imshow videos """
        key = key&0xff 
	# entering hot key mode
        if key == key_map['interrupt_key']:
            self.state = 'interrupt'
            key = cv2.waitKey()&0xff

	if self.state == 'interrupt':
            # must be valid key
            if key == (-1&0xff):
                key = cv2.waitKey()&0xff
            # single step                 
            if key == key_map['next_step_key']:
                return 0
            # leaving hot key mode
            if key == key_map['continue_key']:
                self.state = 'idle'
                return 0
            # return anything else
            return key
         
        return key    


