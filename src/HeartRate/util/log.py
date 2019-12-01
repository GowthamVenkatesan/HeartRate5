import time

class Log():

    def __init__(self, LOG_TAG):
        self.LOG_TAG = LOG_TAG
    
    def log(self, message):
        print(f"{self.LOG_TAG}: {time.asctime(time.localtime(time.time()))}: {message}")