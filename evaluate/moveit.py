import cv2

#for it in range(1, 51):
#    I = cv2.imread("/home/grogan/Storage/CSCE633_Project_data/clear/%s.jpg"%str(it).zfill(4))
#    savepath = "img/{0}.png".format(it)
#    cv2.imwrite(savepath, I)

for it in range(1, 11):
    I = cv2.imread("/home/grogan/Storage/CSCE633_Project_data/train/%s_0.85_0.06.jpg"%str(it).zfill(4))
    savepath = "img/{0}.png".format(it)
    cv2.imwrite(savepath, I)
for it in range(11, 21):
    I = cv2.imread("/home/grogan/Storage/CSCE633_Project_data/train/%s_0.9_0.08.jpg"%str(it).zfill(4))
    savepath = "img/{0}.png".format(it)
    cv2.imwrite(savepath, I)
for it in range(21, 31):
    I = cv2.imread("/home/grogan/Storage/CSCE633_Project_data/train/%s_0.95_0.16.jpg"%str(it).zfill(4))
    savepath = "img/{0}.png".format(it)
    cv2.imwrite(savepath, I)
for it in range(31, 41):
    I = cv2.imread("/home/grogan/Storage/CSCE633_Project_data/train/%s_0.8_0.04.jpg"%str(it).zfill(4))
    savepath = "img/{0}.png".format(it)
    cv2.imwrite(savepath, I)
for it in range(41, 51):
    I = cv2.imread("/home/grogan/Storage/CSCE633_Project_data/train/%s_1_0.2.jpg"%str(it).zfill(4))
    savepath = "img/{0}.png".format(it)
    cv2.imwrite(savepath, I)
