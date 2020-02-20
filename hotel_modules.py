import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(conf_matrix,cats=[],save=False,
                         cmap=plt.cm.Blues):
    n_classes = len(cats)
    
    # Format the class names
    class_names = [c.replace('_',' ') for c in cats]
    
    plt.clf()
    plt.imshow(conf_matrix,cmap=cmap)
    fmt = '.2f'
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            if conf_matrix[i,j] > thresh:
                col = "white"
            else:
                col = 'black'
            
            plt.gca().text(j, i, format(conf_matrix[i, j], fmt),
                    ha="center", va="center",color=col)
    plt.xticks(np.arange(n_classes),class_names,rotation=45,ha='right')
    plt.yticks(np.arange(n_classes),class_names)
    plt.xlim(-0.5,n_classes-0.5)
    plt.ylim(n_classes-0.5,-0.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted')
    plt.tight_layout()
    #
    if save:
        plt.savefig(save,dpi=300)
    plt.show()