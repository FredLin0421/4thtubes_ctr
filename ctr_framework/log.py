import numpy as np
def log(count,multiplier,i,flag,error,detection):
    f= open("collision_log.txt","a+")
    f.write("collision %d\r\n" % (count))
    f.write("multiplier %d\r\n" % (multiplier))
    f.write("i %d\r\n" % (i))
    f.write("flag %d\r\n" % (flag))
    f.write("error %d\r\n" % (error))
    # np.savetxt("collision_log.txt", detection)
    f.close()
    
