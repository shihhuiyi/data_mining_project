# %%
import json
import pandas as pd
import numpy as np
from pandas.core.dtypes.missing import notna

data = pd.read_csv("data.csv")

# %%
data["掛號"] = pd.to_datetime(data["掛號日期"],format="%Y%m%d")

# %%
d = data['掛號日期']
for i in range(len(d)):
    d.iloc[i]=d.iloc[i].replace(d.iloc[i][0:3], str(int(d.iloc[i][0:3]) + 1911))
d.head()
print('-'*20)
d=pd.to_datetime(d,format='%Y/%m/%d')
# %%
#a = "106/07/10"
y,m,d = d.split("/")
# %%
d = data['掛號日期']
# %%
def transform_date(date):
    y,m,d = date.split("/")
    return str(int(y)+1911)+m+d

def transform(data):
    return [transform_date(i) for i in data]

data["掛號"] = pd.to_datetime(transform(data["掛號日期"]))
# %%

df = pd.read_csv("data.csv")
df = df[df["掛號日期"].notna()]
df = df[df["發照日期"].notna()]
df["掛號"] = pd.to_datetime(transform(df["掛號日期"]))
df["發照"] = pd.to_datetime(transform(df["發照日期"]))
df["time"] = (df["發照"]-df["掛號"]).dt.days
# %%
from core.data_loader import DataLoader
import json

config = json.load(open("config.json", "r", encoding="utf-8-sig"))
data = DataLoader(config["data"]["cols"],config["training"]["split"])
# %%
import pandas as pd
df = pd.read_csv("data/data.csv")
df = df.iloc[:,:37]
#df.dropna(how="any",axis=0)
# %%
col = ['掛號日期','起造人',
        '發照日期','地上層數', '地下層數', '建築物高度',
       '總樓地板面積', '工程造價', '建築物用途','戶數', '設計人', '監造人', '承造人', '使用分區', '基地面積', '構造別',
       '建築面積(騎樓)', '建築面積(其他)', '供公眾否']
df= df[col]
#pd.get_dummies(df["供公眾否"])
# %%
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
#labelencoder = LabelEncoder()
#df["供公眾否"] = labelencoder.fit_transform(df["供公眾否"])
#df["供公眾否"] = to_categorical(df["供公眾否"],len(df["供公眾否"]))
# %%
a = to_categorical(df["供公眾否"],len(df["供公眾否"]))
# %%
from core.data_loader import DataLoader
import json
config = json.load(open("config.json","r",encoding="utf-8-sig"))
data = DataLoader(config["data"]["cols"],config["data"]["cat_cols"],config["training"]["split"])
# %%
from featurewiz import featurewiz
target = ["time"]
feature = featurewiz(data.df, target, corr_limit=0.70,verbose=2)

# %%
from core.data_processor import standard
a = standard(data.data_train[:,:-1])
# %%
from sklearn.preprocessing import MinMaxScaler
import numpy as np
a = np.arange(12).reshape(-1,4)
sc = MinMaxScaler()
sc.fit_transform(a)
b = np.arange(8).reshape(-1,4)
c = sc.transform(b)
# %%
import pandas as pd
import numpy as np
a = np.arange(4)
# %%
a = a.sort_values("d")
# %%
import pandas as pd

df = pd.read_csv("data/data.csv")
df = df.describe()
df.to_csv("describe.csv",encoding="utf_8_sig")
# %%
from core.data_loader import DataLoader
import json
import pandas as pd
config = json.load(open("config.json","r",encoding="utf-8-sig"))
data = DataLoader(config["data"]["cols"],config["data"]["cat_cols"],config["training"]["split"])
# %%
