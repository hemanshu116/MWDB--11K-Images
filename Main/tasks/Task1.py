from Main.helper import processFeatureDescriptors


def startTask1():
    featureVector = processFeatureDescriptors()
    # print(len(featureVector))
    print(type(featureVector))


# Uncomment to run task independently
# startTask1()
