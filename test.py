import pandas
import pickle
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
loaded_model = pickle.load(open('prediction_model.pkl', 'rb'))
input_data = [[3180,4,2,0,0,0,0,0,1]]
scaled_input = sc.fit_transform(input_data)
prediction = loaded_model.predict(scaled_input)
print(prediction)
