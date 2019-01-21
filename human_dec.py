import numpy as np
import cv2
import math
import os


class HOG:

    def imagesNames(self):
        #loads up the images
        images = []
        positiveImg = os.listdir('./Train_Positive')
        negativeImg = os.listdir('./Train_Negative')
        for i in range(10):
            images.append('/Train_Positive/' + positiveImg[i])
            images.append('/Train_Negative/' + negativeImg[i])

        expectedOutput = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])

        return images, expectedOutput

    def caller(self, images, expectedOutput):
        img_pross = ImgPreProcessing()
        NNtraining = NNtrain()
        NNtest = NeuralNetworkTest()
        counter = 0
        flag = 0
        while NNtrain.errorChange > 0.0005 or NNtrain.epochCounter <= 50:
            for i in range(20):
                if counter < 20:
                    img = cv2.imread('.' + images[i])
                    image = img_pross.imageGrayscale(img)
                    gradientMagnitude, gradientAngle = img_pross. prewitt(image)
                    cellHistogram = img_pross.hogCell(gradientMagnitude, gradientAngle)
                    tempHistogram = img_pross.hogBlock(cellHistogram)
                    
                    #print(tempHistogram.tolist())
                    #print(" ")
                    if flag == 0:
                        flag = 1
                        blockHistogram = np.empty((20, ImgPreProcessing.sizeOfInput))
                    blockHistogram[i][:] = tempHistogram
                    counter = counter + 1
                observedOutput = NNtraining.neuralTraining(blockHistogram[i][:])
                NNtraining.backPropagation(blockHistogram[i][:], observedOutput, expectedOutput[i])



        testImages = []
        positiveTest = os.listdir('./Test_Positive')
        negativeTest = os.listdir('./Test_Neg')
        for i in range(5):
            testImages.append('/Test_Positive/' + positiveTest[i])
            testImages.append('/Test_Neg/' + negativeTest[i])
        expectedOutputTest = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        correctPrediction = 0
        print(testImages)
        wrongPrediction = 0
        for i in range(10):
            imgTest = cv2.imread('.' + testImages[i])
            imageTest = img_pross.imageGrayscale(imgTest)
            gradientMagnitudeTest, gradientAngleTest = img_pross.prewitt(imageTest)
            cellHistogramTest = img_pross.hogCell(gradientMagnitudeTest, gradientAngleTest)
            blockHistogramTest = img_pross.hogBlock(cellHistogramTest)
            observedOutputTest = NNtest.testImage(blockHistogramTest)
            print('Image ->', (i + 1))
            print(testImages[i])
            print('Observed Output ->', observedOutputTest)
            print('Expected output ->', expectedOutputTest[i])

           

            print('Error : ', abs(observedOutputTest - expectedOutputTest[i]))

            if observedOutputTest >= 0.5:
                print('Human is present.')
            else:
                print('Human is not present.')
            print()

            if 0 < observedOutputTest - expectedOutputTest[i] < 0.5:
                correctPrediction += 1
            if 0 < expectedOutputTest[i] - observedOutputTest < 0.5:
                correctPrediction += 1
            if 0.5 < observedOutputTest - expectedOutputTest[i] < 1:
                wrongPrediction += 1
            if 0.5 < expectedOutputTest[i] - observedOutputTest < 1:
                wrongPrediction += 1
        print("Correct Predictions :", correctPrediction)
        print("Incorrect Predictions :", wrongPrediction)


class ImgPreProcessing:

    sizeOfInput = 0
#Converts RGB image to grayscale
    def imageGrayscale(self, rgb):
        return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#Applies the perwitt edge operator 

    def prewitt(self, img):
    
        Hx = np.array([[-1, 0, 1],
                                      [-1, 0, 1],
                                      [-1, 0, 1]])

        Hy = np.array([[1, 1, 1],
                                    [0, 0, 0],
                                    [-1, -1, -1]])

        height = img.shape[0]
        width = img.shape[1]

#Initialize emoty array for Hx and Hy
        horizontalGradient = np.zeros((height, width))
        verticalGradient = np.zeros((height, width))

        for i in range(1, height - 1, 1):
            for j in range(1, width - 1, 1):
                x = 0
                for k in range(3):
                    for l in range(3):
                        x = x + ((img[i - 1 + k][j - 1 + l]) * Hx[k][l])
                horizontalGradient[i][j] = x / 3

        for i in range(1, height - 1, 1):
            for j in range(1, width - 1, 1):
                x = 0
                for k in range(3):
                    for l in range(3):
                        x = x + (img[i - 1 + k][j - 1 + l] * Hy[k][l])
                verticalGradient[i][j] = x / 3

        gradientAngle = np.zeros((height, width))

        for i in range(1, height - 1, 1):
            for j in range(1, width - 1, 1):
                if horizontalGradient[i][j] == 0 and verticalGradient[i][j] == 0:
                    gradientAngle[i][j] = 0
                elif horizontalGradient[i][j] == 0 and verticalGradient[i][j] != 0:
                    gradientAngle[i][j] = 90
                else:
                    x = math.degrees(math.atan(verticalGradient[i][j] / horizontalGradient[i][j]))
                    if x < 0:
                        x = 360 + x
                    if x >= 170 or x < 350:
                        x = x - 180
                    gradientAngle[i][j] = x

        gradientMagnitude = np.zeros((height, width), dtype='int')

        for i in range(1, height - 1, 1):
            for j in range(1, width - 1, 1):
                x = math.pow(horizontalGradient[i][j], 2) + math.pow(verticalGradient[i][j], 2)
                gradientMagnitude[i][j] = int(round(math.sqrt(x / 2)))
        return gradientAngle, gradientMagnitude

    def hogCell(self, gradientAngle, gradientMagnitude):
#computes the Hog Cell
        height = gradientAngle.shape[0]
        width = gradientAngle.shape[1]

        cellHistogram = np.zeros((int(height / 8), int(width * 9 / 8)))

        tempHist = np.zeros((1, 9))
#Puts the angle in its corresponding bin
        for i in range(0, height - 7, 8):
            for j in range(0, width - 7, 8):
                tempHist = tempHist * 0
                for k in range(8):
                    for l in range(8):
                        angle = gradientAngle[i + k][j + l]
                        if -10 <= angle < 0:
                            dist = 0 - angle
                            tempHist[0][0] = tempHist[0][0] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
                            tempHist[0][8] = tempHist[0][8] + dist * gradientMagnitude[i + k][j + l] / 20
                        elif 0 <= angle < 20:
                            dist = angle
                            tempHist[0][0] = tempHist[0][0] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
                            tempHist[0][1] = tempHist[0][1] + dist * gradientMagnitude[i + k][j + l] / 20
                        elif 20 <= angle < 40:
                            dist = angle - 20
                            tempHist[0][1] = tempHist[0][1] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
                            tempHist[0][2] = tempHist[0][2] + dist * gradientMagnitude[i + k][j + l] / 20
                        elif 40 <= angle < 60:
                            dist = angle - 40
                            tempHist[0][2] = tempHist[0][2] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
                            tempHist[0][3] = tempHist[0][3] + dist * gradientMagnitude[i + k][j + l] / 20
                        elif 60 <= angle < 80:
                            dist = angle - 60
                            tempHist[0][3] = tempHist[0][3] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
                            tempHist[0][4] = tempHist[0][4] + dist * gradientMagnitude[i + k][j + l] / 20
                        elif 80 <= angle < 100:
                            dist = angle - 80
                            tempHist[0][4] = tempHist[0][4] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
                            tempHist[0][5] = tempHist[0][5] + dist * gradientMagnitude[i + k][j + l] / 20
                        elif 100 <= angle < 120:
                            dist = angle - 100
                            tempHist[0][5] = tempHist[0][5] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
                            tempHist[0][6] = tempHist[0][6] + dist * gradientMagnitude[i + k][j + l] / 20
                        elif 120 <= angle < 140:
                            dist = angle - 120
                            tempHist[0][6] = tempHist[0][6] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
                            tempHist[0][7] = tempHist[0][7] + dist * gradientMagnitude[i + k][j + l] / 20
                        elif 140 <= angle < 160:
                            dist = angle - 140
                            tempHist[0][7] = tempHist[0][7] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
                            tempHist[0][8] = tempHist[0][8] + dist * gradientMagnitude[i + k][j + l] / 20
                        elif 160 <= angle < 170:
                            dist = angle - 160
                            tempHist[0][8] = tempHist[0][8] + (20 - dist) * gradientMagnitude[i + k][j + l] / 20
                            tempHist[0][0] = tempHist[0][0] + dist * gradientMagnitude[i + k][j + l] / 20
                cellHistogram[int(i / 8)][int(j * 9 / 8):int(j * 9 / 8 + 9)] = tempHist
        return cellHistogram

    def hogBlock(self, cellHistogram):
#computes the Hog Block
        height = cellHistogram.shape[0]
        width = cellHistogram.shape[1]

        blockHistogram = np.empty((int(height - 1), int((width / 9 - 1) * 36)))
        tempHistogram = np.zeros((1, 36))

        for i in range(0, height - 1, 1):
            for j in range(0, width - 17, 9):
                l2Norm = 0
                for k in range(2):
                    for l in range(18):
                        l2Norm = l2Norm + math.pow(cellHistogram[i + k][j + l], 2)
                l2Norm = math.sqrt(l2Norm)
                x = 0
                for k in range(2):
                    for l in range(18):
                        if l2Norm == 0:
                            tempHistogram[0][x] = 0
                        else:
                            tempHistogram[0][x] = cellHistogram[i + k][j + l] / l2Norm
                        x = x + 1
                blockHistogram[i][int(j * 36 / 9):int(j * 36 / 9 + 36)] = tempHistogram
        blockHistogram = blockHistogram.flatten()
        ImgPreProcessing.sizeOfInput = blockHistogram.shape[0]
        return blockHistogram


def reLu(num):
#Used to keep all negatives at 0
    if num <= 0:
        return 0
    else:
        return num


def reLuDeriv(num):
    if num <= 0:
        return 0
    else:
        return 1


class NNtrain:
    weights1 = None
    weights2 = None
    sizeOfHidden = 0
    hiddenInput = None
    flag = 0
    counter = 0
    sqError = 0
    epochCounter = 0
    prevError = None
    errorChange = 0

    def neuralTraining(self, blockHistogram):
        #trains the neural network
        if NNtrain.flag == 0:
            NNtrain.sizeOfHidden = int(input("Number of hidden layers : "))
            #Asks user for the number of hindden layers 
            NNtrain.weights1 = np.random.randn(NNtrain.sizeOfHidden, ImgPreProcessing
            .sizeOfInput)
            NNtrain.weights1 = np.multiply(NNtrain.weights1, math.sqrt(2 / int(ImgPreProcessing
            .sizeOfInput + NNtrain.sizeOfHidden)))
            NNtrain.weights2 = np.random.randn(NNtrain.sizeOfHidden)
            NNtrain.weights2 = NNtrain.weights2 * math.sqrt(1 / int(NNtrain.sizeOfHidden))
            NNtrain.flag = 1

        NNtrain.hiddenInput = np.empty(NNtrain.sizeOfHidden)

        NNtrain.hiddenInput = np.matmul(NNtrain.weights1, blockHistogram)
        sigmoidInput = np.matmul(list(map(reLu, NNtrain.hiddenInput)), NNtrain.weights2)
        observedOutput = 1 / (1 + np.exp(-sigmoidInput))
        return observedOutput

    def backPropagation(self, blockHistogram, observedOutput, expectedOutput):

        err = expectedOutput - observedOutput
        x = (-err) * observedOutput * (1 - observedOutput)

        a = np.multiply(NNtrain.weights2, x)
        b = np.multiply(a, list(map(reLuDeriv, NNtrain.hiddenInput)))
        change1 = np.matmul(b.reshape(NNtrain.sizeOfHidden, 1), blockHistogram.reshape(1, ImgPreProcessing
        .sizeOfInput))
        NNtrain.weights1 = np.subtract(NNtrain.weights1, np.multiply(change1, 0.1))

        change2 = np.multiply(list(map(reLu, NNtrain.hiddenInput)), 0.1 * x)
        NNtrain.weights2 = np.subtract(NNtrain.weights2, change2)

        NNtrain.sqError = NNtrain.sqError + math.pow(err, 2)
        NNtrain.counter = NNtrain.counter + 1
        if NNtrain.counter == 20:
            NNtrain.counter = 0
            NNtrain.epochCounter = NNtrain.epochCounter + 1
            if NNtrain.epochCounter == 1:
                NNtrain.prevError = NNtrain.sqError
            else:
                NNtrain.errorChange = NNtrain.prevError - NNtrain.sqError
                NNtrain.prevError = NNtrain.sqError
            NNtrain.sqError = 0


class NeuralNetworkTest:
    #Tests the unknown images for presence of humans
    def testImage(self, blockHistogram):
        hiddenInput = np.matmul(NNtrain.weights1, blockHistogram)
        sigmoidInput = np.matmul(list(map(reLu, hiddenInput)), NNtrain.weights2)
        observedOutput = 1 / (1 + np.exp(-sigmoidInput))
        return observedOutput


def main():
    hog = HOG()
    a, b = hog.imagesNames()
    hog.caller(a, b)

main()