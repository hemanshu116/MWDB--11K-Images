'''

'''

def startTask5():
    inps = get_usr_inp()


def get_usr_inp():
    print('Choose from any one of the following Feature Models')
    print('1. Color Moments\n2. Local Binary Pattern\n3. Histogram of Oriented Gradients\n4. Scale-invarient Feature Transform')
    feat_ch = input('Enter your choice: ')

    print('Choose from any one of the following reducer technique: ')
    print('1. PCA\n2. SVD\n3. NMF\n4. LDA')
    redux_ch = input('Enter your choice: ')

    k = input('Enter k (number of latent semantics): ')

    print('Choose from any one of the following Labels: ')
    print('1. left-hand\n2. right-hand\n3. dorsal\n4. palmar\n5. with accessories\n6. without accesories\n7. Male\n8. Female')
    label_ch = input('Enter your choice: ')

    return {'feature': feat_ch, 'reduction': redux_ch, 'k': k, 'label': label_ch}