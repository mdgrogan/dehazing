lines = [line.rstrip('\n').split(' ') for line in open("hey.txt", "r")]

# lines[0][X] = header
# lines[X][0] = image 
# lines[X][1] = mean_lum
# lines[X][2] = var_lum
# lines[X][3] = entropy
# lines[X][4] = entropy R
# lines[X][5] = entropy G
# lines[X][6] = entropy B
# lines[X][7] = clip
# lines[X][8] = tile 
# lines[X][9] = PSNR
# lines[X][10] = SSIM 
# lines[X][11] = pre-PSNR
# lines[X][12] = pre-SSIM 

bestPSNRs = []
bestSSIMs = []

for i in range(1,len(lines)):
    if lines[i][0] != lines[i-1][0]:
        maxPSNR = i
        maxSSIM = i

    if float(lines[i][9]) > float(lines[maxPSNR][9]):
        maxPSNR = i
    if float(lines[i][10]) > float(lines[maxPSNR][10]):
        maxSSIM = i

    if i+1 < len(lines):
        if lines[i][0] != lines[i+1][0]:
            bestPSNRs.append(lines[maxPSNR])
            bestSSIMs.append(lines[maxSSIM])
    if i+1 == len(lines):
        bestPSNRs.append(lines[maxPSNR])
        bestSSIMs.append(lines[maxSSIM])

f = open("best_PSNRs.txt", "w+")
f.write(" ".join(elem for elem in lines[0])+'\n')
for l in bestPSNRs:
    f.write(" ".join(elem for elem in l)+'\n')
f.close()

f = open("best_SSIMs.txt", "w+")
f.write(" ".join(elem for elem in lines[0])+'\n')
for l in bestSSIMs:
    f.write(" ".join(elem for elem in l)+'\n')
f.close()
