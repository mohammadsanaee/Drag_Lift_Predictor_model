import xgboost as xgb
from sklearn.metrics import r2_score

# Load the model
model = xgb.XGBRegressor()
model.load_model('Cl_Cd_prediction _model.model')

#getting the inputs from the user
print("Enter values for the following input features:")
c = float(input("c: "))
b = float(input("b: "))
A_over_C = float(input("A/C: "))
lambda_over_C = float(input("λ/C: "))
h_over_c = float(input("h/c: "))
alpha = float(input("Alpha: "))
Re = float(input("Re: "))
sample_input = [[c, b, A_over_C, lambda_over_C, h_over_c, alpha, Re]]
dtest = xgb.DMatrix(sample_input, feature_names=['c', 'b', 'A/C', 'λ/C', 'h/c', 'Alpha', 'Re'])
csr_test = dtest.get_data()

#make the prediction
predictions = model.predict(csr_test)
print(f"Predicted C_L: {predictions[0][0]:.2f}, Predicted C_D: {predictions[0][1]:.2f}")
