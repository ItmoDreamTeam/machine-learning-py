from sklearn.linear_model import LinearRegression

from linearregression.apartment import *

apartments = ApartmentCsvReader.read_apartments()
print(apartments)

X = [[a.area, a.rooms] for a in apartments]
y = [a.price for a in apartments]
print('X = %s\nY = %s' % (X, y))

lr = LinearRegression(fit_intercept=True, normalize=True)
lr.fit(X, y)
print("Coeff = %s" % [lr.intercept_, lr.coef_])
print("MSE = %e" % (sum((lr.predict(X) - y) ** 2) / len(X)))

while True:
    print("Input apartment: <area> <rooms>")
    area, rooms = map(float, input().split())
    print("Predicted price = %f" % lr.predict([[area, rooms]])[0])
