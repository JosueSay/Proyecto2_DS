from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def jaccardTokens(a: str, b: str) -> float:
    sa, sb = set(str(a).split()), set(str(b).split())
    if not sa and not sb:
        return 1.0  # si ambos strings están vacíos, considerarlos idénticos
    return len(sa & sb) / max(len(sa | sb), 1)  # intersección / unión

def cosineTfidf_pairs(a_list, b_list):
    vect = TfidfVectorizer(min_df=1)  # crear vectorizador tf-idf
    X = vect.fit_transform(list(a_list) + list(b_list))  # transformar todos los textos
    XA = X[:len(a_list)]  # separar matriz del primer conjunto
    XB = X[len(a_list):]  # separar matriz del segundo conjunto
    num = (XA.multiply(XB)).sum(axis=1).A1  # numerador del coseno
    den = (np.sqrt((XA.multiply(XA)).sum(axis=1).A1) * np.sqrt((XB.multiply(XB)).sum(axis=1).A1) + 1e-12)  # denominador del coseno
    return (num / den).tolist()  # similitud coseno como lista
