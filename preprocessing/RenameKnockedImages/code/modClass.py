import os

# Read in the file
#with open('/home/antke/Desktop/train/labels/IMG_20211108_145300_jpg.rf.30d514f9d08f87b26d69605927a1e76c.txt', 'r') as file :
#  filedata = file.read()

# Replace the target string
  #all_lines = [[int(num) for num in line.split()] for line in filedata]
  
#filedata = filedata.replace('ram', 'abcd')

# Write the file out again
#with open('file.txt', 'w') as file:
#  file.write(filedata)
import os
source_path = "/home/antke/Desktop/train/labels/"
destination_path = "/home/antke/Desktop/train/labels_mod/"
for i, filename in enumerate(os.listdir(source_path)):
    with open(source_path + filename, 'r') as fobj:
        #filedata = fobj.read()
        f = open(destination_path + filename,'w+')
        old_line = ""
        new_line = ""
        for line in fobj:
            numbers = [num for num in line.split()]
            print(numbers[0])
            old_line = str(numbers[0]) + " " + str(numbers[1]) + " " + str(numbers[2]) + " " + str(numbers[3]) + " " + str(numbers[4])
            if(numbers[0]==str(0)):
                new_line = "1" + " " + str(numbers[1]) + " " + str(numbers[2]) + " " + str(numbers[3]) + " " + str(numbers[4])
                print(new_line)
                #1
            if(numbers[0]==str(1)):
                new_line = "5" + " " + str(numbers[1]) + " " + str(numbers[2]) + " " + str(numbers[3]) + " " + str(numbers[4])
                print(new_line)
                #5
            if(numbers[0]==str(2)):
                new_line = "9" + " " + str(numbers[1]) + " " + str(numbers[2]) + " " + str(numbers[3]) + " " + str(numbers[4])
                print(new_line)
                #9
            f.write(str(new_line) + '\n')
        fobj.close()
