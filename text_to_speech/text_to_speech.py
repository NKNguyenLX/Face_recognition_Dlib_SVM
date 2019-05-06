# import pyttsx
# engine = pyttsx.init()
# engine.say('nguyen khoi nguyen')
# engine.runAndWait()

from gtts import gTTS
import os
tts = gTTS(text='bui thi anh thu', lang='vi')
tts.save("good.mp3")
os.system("mpg321 good.mp3")