"""Maintained minimal recipe example used in the MLOSS manuscript."""
from sksurv.datasets import load_whas500
from sklearn.model_selection import train_test_split
from texgisa_survival.recipes import SurvivalRecipe

X,y=load_whas500();X=X.astype(float)
a,b=train_test_split(range(len(X)),test_size=.25,random_state=271,stratify=y['fstat'])
model=SurvivalRecipe('RefineCDF',random_state=271)
model.fit(X.iloc[a],y['lenfol'][a],y['fstat'][a])
risk=model.predict_risk(X.iloc[b])
survival=model.predict_survival(X.iloc[b],times=[30.,180.,365.])
point_time=model.predict_time(X.iloc[b])
print(risk.shape,survival.shape,point_time.shape)
