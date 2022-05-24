import pickle


def iris_elasticnet_model():

    import pandas as pd
    from sklearn.linear_model import ElasticNet, LinearRegression

    df = pd.read_csv(
        "https://raw.githubusercontent.com/LineaLabs/lineapy/main/examples/tutorials/data/iris.csv"
    )
    color_map = {"Setosa": 0, "Versicolor": 1, "Virginica": 2}
    df["variety_color"] = df["variety"].map(color_map)
    model2 = ElasticNet()
    model2.fit(
        X=df[["petal.width", "variety_color"]],
        y=df["sepal.width"],
    )
    pickle.dump(model2, open("/home/jovyan/.lineapy/linea_pickles/215Tsod", "wb"))
