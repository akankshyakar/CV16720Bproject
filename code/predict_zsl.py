
import sys
import numpy as np
from PIL import Image

from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize
from util import *
from zsl_train import *



WORD2VECPATH    = "../data/class_vectors.npy"
MODELPATH       = "../model/"

def main(img_file):

    # if len(argv) != 1:
    #     print("Give input image to compute")
    #     exit()

    # READ IMAGE
    # img_file = argv[0]
    img= Image.open(img_file)

    img_feature = get_features(img_file)
    # print(img_feature.shape)
    img_feature = normalize(img_feature, norm='l2')
    # print(img_feature[0])
    # print("iam anew")
    mymodel = MyNet().to(device)
    # print(mymodel)
    optimizer = optim.Adam(mymodel.parameters(), lr=5e-5)
    checkpoint=load_checkpoint('model/zsl_-7000.pth', mymodel, optimizer)
    modulelist = list(mymodel.modules())
    model = nn.Sequential(*modulelist[1:-3])
    # print(model)
    img_feature=torch.tensor(img_feature).float()
    # print(img_feature[0])
    model.eval() 
    pred_zsl = model(img_feature).detach().numpy()
    # assert(2==3)

    
    # print(pred_zsl[0])

    class_vectors= sorted(np.load(WORD2VECPATH,allow_pickle=True), key=lambda x: x[0])
    # print(class_vectors)
    classnames, vectors = zip(*class_vectors)
    classnames= list(classnames)
    # print(classnames)
    vectors= np.asarray(vectors, dtype=np.float)
    # print(vectors.shape)
    tree= KDTree(vectors)

    dist, index= tree.query(pred_zsl, k=5)
    # print(dist, index)
    pred_labels= [classnames[idx] for idx in index[0]]

    # PRINT RESULT
    print(img_file)
    print("--- Top-5 Prediction ---")
    for i, classname in enumerate(pred_labels):
        print("%d- %s" %(i+1, classname))
    print()
    return

if __name__ == '__main__':
    # main(sys.argv[1:])



    for f in os.listdir("../test_images"):
        # print(f)
        main("../test_images/"+f)
