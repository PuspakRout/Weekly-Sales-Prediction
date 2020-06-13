import schedule
from codes import *
import time

schedule.every().monday.do(refitting)

while True:
    schedule.run_pending()
    time.sleep(1)