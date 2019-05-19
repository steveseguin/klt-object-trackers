
import serial
import time

class ServoController:
    def __init__(self):
        usbPort = 'COM5'
        self.sc = serial.Serial(usbPort, timeout=1)

    def closeServo(self):
        self.sc.close()

    def setAngle(self, n, angle):
        if angle > 180 or angle <0:
           angle=90
        byteone=int(254*angle/180)
        bud=chr(0xFF)+chr(n)+chr(byteone)
        self.sc.write(bud)

    def setPosition(self, servo, position):
        position = position * 4
        poslo = (position & 0x7f)
        poshi = (position >> 7) & 0x7f
        chan  = servo &0x7f
        data =  chr(0xaa) + chr(0x0c) + chr(0x04) + chr(chan) + chr(poslo) + chr(poshi)
        self.sc.write(data)

    def getPosition(self, servo):
        chan  = servo &0x7f
        data =  chr(0xaa) + chr(0x0c) + chr(0x10) + chr(chan)
        self.sc.write(data)
        w1 = ord(self.sc.read())
        w2 = ord(self.sc.read())
        return w1, w2

    def getErrors(self):
        data =  chr(0xaa) + chr(0x0c) + chr(0x21)
        self.sc.write(data)
        w1 = ord(self.sc.read())
        w2 = ord(self.sc.read())
        return w1, w2

    def triggerScript(self, subNumber):
        data =  chr(0xaa) + chr(0x0c) + chr(0x27) + chr(0)
        self.sc.write(data)
		
servos = ServoController()

##f = open("data2.txt", "w")


while (1==1):
	output = servos.getPosition(0)
	value = output[0]+output[1]*255
	if value>650:
		print "!",
	else:
		print ".",
		##f.write( str(output[0]+output[1]*255) + "\n"  )
##f.close()