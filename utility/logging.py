import os
from datetime import datetime

class Logger():
    def __init__(self, filename, is_debug, path='./logs/'):
        self.filename = filename + '_improvment_version'
        self.path = os.path.join(os.path.join(path, self.filename)).replace(" ", "_")
        self.log_ = not is_debug
        os.makedirs(self.path, exist_ok=True)

    def logging(self, s):
        s = str(s)
        print(datetime.now().strftime('%Y-%m-%d %H:%M: '), s)
        if self.log_:
            with open(os.path.join(self.path, '.log,'), 'a+') as f_log:
                f_log.write(str(datetime.now().strftime('%Y-%m-%d %H:%M:  ')) + s + '\n')



