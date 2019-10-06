from .predict_sentiment import predict as _predict

def predict(sentence):
	r = _predict([sentence])[0]
	if r == 0:
		#positive
		return 1
	elif r == 1:
		#strongly positive
		return 2
	elif r == 2:
		#negative
		return -1
	else:
		#strongly negative
		return -2