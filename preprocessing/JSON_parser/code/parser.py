import json
import os

# rename knocked cones class name

'''source_path = "../../../Desktop/FSOCO_nowatermark/ann/"
destination_path = ""
for i, filename in enumerate(os.listdir(source_path)):
    with open(source_path + filename) as f:
        data = json.load(f)
        for objects in data['objects']:
            #for tags in objects['tags']:
            if objects['name']=='knocked_over':
                objects['classId']=round(objects['classId']/2)
                objects['classTitle']=str(objects['classTitle']) + '_knocked'
                print('class updated \n')

    with open(source_path + filename, "w") as f:
        json.dump(data, f)'''



'''# shift bounding boxes coordinates after removing watermarks
source_path = "../../../../..//media/antke/Elements1/TesiMagistrale/other/datasets_JSON/FSOCO_nowatermark/"
#for team in os.listdir(source_path):
for i, filename in enumerate(os.listdir(source_path)):
    with open(source_path + '/'  + filename) as f:
        data = json.load(f)
        for objects in data['objects']:
            for tags in objects['tags']:
                if tags['name']=='knocked_over':
                    objects['classId']=round(objects['classId']/2)
                    objects['classTitle']=str(objects['classTitle']) + '_knocked'
                    print('class updated \n')
                    continue
            for points in objects['points']['exterior']:
                    points[0] -= 140
                    points[1] -= 140
                    print('Bounding boxes updated')

        with open('../../../Desktop/total/upd_ann/' + filename, "w") as f:
            json.dump(data, f)'''


# change large orange in orange class
source_path = "../../../Desktop/total/ann/"
#for team in os.listdir(source_path):
blu=0
yellow=0
orange=0
for i, filename in enumerate(os.listdir(source_path)):
    with open(source_path + '/'  + filename) as f:
        data = json.load(f)
        for objects in data['objects']:
            if (int(objects['classId']) == 2744363):
                blu += 1
            if (int(objects['classId']) == 2744368):
                yellow += 1
            if (int(objects['classId']) == 2744364):
                orange += 1
            if(int(objects['classId']) == 2744365):
                objects['classId']=str(2744364)
                objects['classTitle']="orange_cone"
                orange += 1
                    

        with open('../../../Desktop/total/upd_ann/' + filename, "w") as f:
            json.dump(data, f)
    
print("blu: " + str(blu) + '\n')
print("yellow: " + str(yellow) + '\n')
print("orange: " + str(orange) + '\n')



'''source_path = "../../../../..//media/antke/Elements1/TesiMagistrale/other/datasets_JSON/FSOCO_nowatermark/"
for team in os.listdir(source_path):
    for i, filename in enumerate(os.listdir(source_path + team + '/ann/')):
        with open(source_path + team + '/ann/' + filename) as f:
            data = json.load(f)
            for objects in data['objects']:
                for tags in objects['tags']:
                    if tags['name']=='knocked_over':
                        objects['classId']=round(objects['classId']/2)
                        objects['classTitle']=str(objects['classTitle']) + '_knocked'
                        print('class updated \n')
                        continue
                for points in objects['points']['exterior']:
                        points[0] -= 140
                        points[1] -= 140
                        print('Bounding boxes updated - ' + str(team))

        with open('../../../../..//media/antke/Elements1/TesiMagistrale/other/datasets_JSON/FSOCO_nowatermark/' + team + '/new_ann/' + filename, "w") as f:
            json.dump(data, f)'''