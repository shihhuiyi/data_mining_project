from sklearn.preprocessing import StandardScaler

def transform_date(date):
    y,m,d = date.split("/")
    return str(int(y)+1911)+m+d

def transform(data):
    return [transform_date(i) for i in data]


def standard(self,data):
    sc = StandardScaler()
    sc.fit_transform(self.data_train[:,:-1])
    data = sc.transform(data)
    return data