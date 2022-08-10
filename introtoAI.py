from sklearn import svm

input_data = [
  [5.1, 3.5, 1.4, 0.2],
  [4.9, 3.0, 1.4, 0.2],
  [4.6, 3.1, 1.5, 0.2],
  [7.0,	3.2, 4.7,	1.4],
  [6.4, 3.2, 4.5,	1.5],
  [6.9,	3.1, 4.9, 1.5],
  [6.3,	3.3, 6.0,	2.5],
  [5.8,	2.7, 5.1,	1.9],
  [7.1,	3.0, 5.9,	2.1]
    ]

output_data = [1, 1, 1, 2, 2, 2, 3, 3, 3]
#select right model
model = svm.SVC()
#training
model.fit(input_data, output_data)
#prediction (application)

print(model.predict([
  [5.6,	2.5, 5.0,	1.7],
  [7.0,	3.2, 5.8,	2.3]
]))

