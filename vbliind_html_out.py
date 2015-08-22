import sys
import os 
import re
import numpy as np


#first calculate avg. of brisque scores
brisqueAvgs = []
# will use this to label columns in table for both brisque and vbliinds.
brisqueNames = [f for f in os.listdir('.') if re.match(r'.*BRISQUE.*', f)]
#print "brisqueNames"
#print brisqueNames
brisqueCounter = 0
for name in brisqueNames:
    open_file = open(name, 'r')
    avg = 0
    for line in open_file:
        # 'line' is a line in your file
        # We make sure that 'line' is non-empty
        if line:
            avg += float(line)
            brisqueCounter+=1
    # When finishing reading the file, we close it    
    if (brisqueCounter !=  0):
        brisqueAvgs.append(str(avg/brisqueCounter))
    brisqueCounter = 0
    open_file.close()


vTotalNames = [f for f in os.listdir('.') if re.match(r'vscore*', f)]
vTotalScores = []
vTotalCounter = 0
for name in vTotalNames:
    open_file = open(name, 'r')
    for line in open_file:
        # 'line' is a line in your file
        # We make sure that 'line' is non-empty
        if line:
            vTotalScores.append(line)
            vTotalCounter+=1
    # When finishing reading the file, we close it    
    open_file.close()

vScoreComponentFile = "matlab_scores" 
# Opening file for reading. The path of the file is given in argument
open_file = open(vScoreComponentFile, 'r')
vScoreComponents = []
for line in open_file:
    # 'line' is a line in your file
    if line:
        vScoreComponents.append(line)

# When finishing reading the file, we close it    
open_file.close()

# Now that all the data inside the file has been read, and store in our arrays
#    we have to use it to create / print it in an HTML table

#list of avgs of vScoreComponent 
spatialNaturalness = []
for i in range(0,vTotalCounter):
    #vScoreComponents[i*15+8] = np.array(vScoreComponents[i*15+8]).tolist()
    #vScoreComponents[i*15+8]])/5
    templist = []
    templist = (np.array(vScoreComponents[i*15+1])).tolist()
    newlist = []
    #print (np.array(vScoreComponents[i*15+1].strip('[]'))).tolist()
    #np.array(np.matrix(s.strip('[]'))
    #print templist.strip('[]').split()
    for num in templist.strip('[').strip(']').split():
        newlist.append(float(num.strip('[]')))
    spatialNaturalness.append(str(reduce(lambda x,y:x+y,newlist,0)))
#    print "newlist"
#    print newlist
#    print i
#    print type(templist)
#    print eval(templist)
#    print reduce(lambda x, y: float(x)+float(y),templist,0.0)
    #spatialNaturalness.append(5)
    #print vScoreComponents[i*15+1]

shapeParameterRatios = []
for i in range(0,vTotalCounter):
    #vScoreComponents[i*15+8] = np.array(vScoreComponents[i*15+8]).tolist()
    #vScoreComponents[i*15+8] = np.array(vScoreComponents[i*15+8]).tolist()
    #vScoreComponents[i*15+8]])/5
    templist = []
    templist = (np.array(vScoreComponents[i*15+1])).tolist()
    newlist = []
    #print (np.array(vScoreComponents[i*15+1].strip('[]'))).tolist()
    #np.array(np.matrix(s.strip('[]'))
    #print templist.strip('[]').split()
    for num in templist.strip('[').strip(']').split():
        newlist.append(float(num.strip('[]')))
    shapeParameterRatios.append(str(reduce(lambda x,y:x+y,newlist,0)))
    #shapeParameterRatios.append(np.array(vScoreComponents[i*15+8]).tolist())
    #print vScoreComponents[i*15+8]


# Print opening HTML tags -------------------------
print "<htm><body><table>"
# Print the content of the table, line by line ----
print "<tr><td>Name</td><td>Brisque Score </td><td>videoBliinds Score</td><td>Spatial Naturalness</td><td>DC Feature 1</td><td>DC Feature 2</td><td>NVS Shape-Parameter ratios</td><td>Coherency Measure</td><td>Global Motion Measure</td></tr>"
for i in range(0, vTotalCounter):
        print "<tr><td>"+brisqueNames[i]+"</td><td>"+brisqueAvgs[i]+"</td><td>"+vTotalScores[i]+"</td><td>"+spatialNaturalness[i] + "</td><td>" + vScoreComponents[i*15+4] + "</td><td>"+vScoreComponents[i*15+6]+"</td><td>" + shapeParameterRatios[i] + "</td><td>"+ vScoreComponents[i*15+11]+"</td><td>"+vScoreComponents[i*15+13]+"</td></tr>"
        #print "<tr><td>"+brisqueNames[i]+"</td><td>"+brisqueAvgs[i]+"</td><td>"+vTotalScores[i]+"</td><td>"+ vScoreComponents[i*15+1] + "</td><td>" + vScoreComponents[i*15+3] + "</td><td>" + vScoreComponents[i*15+5] + "</td></tr>"
 
# Print closing HTML tags -------------------------
print "</table></body></html>"
