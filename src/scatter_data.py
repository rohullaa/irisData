#importing libraries

def main():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn import datasets
    from sklearn import svm

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
                names= ["SepalLength","SepalWidth","PetalLength","PetalWidth","Species"])


    fig,axes = plt.subplots(nrows = 2, ncols = 2, figsize=(10,10))
    def plotting(ax,sLength, sWidth):
        colors = ['red','orange','blue']
        species = ['Iris-setosa','Iris-versicolor','Iris-virginica']
        for i in range(3):
            x = df[df['Species'] == species[i]]
            ax.scatter(x[sLength],x[sWidth],c=colors[i],label=species[i])
        ax.set_xlabel(sLength)
        ax.set_ylabel(sWidth)
        ax.legend()


    plotting(axes[0][0],"SepalLength","SepalWidth")
    plotting(axes[1][0],"PetalLength","PetalWidth")
    plotting(axes[0][1],"SepalLength","PetalLength")
    plotting(axes[1][1],"SepalWidth","PetalWidth")
    #plt.savefig("scatter_data.png")
    plt.show()
    
    return fig
    

if __name__ == "__main__":
    main()
