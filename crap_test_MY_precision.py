import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

X_train = np.array(["so bad",
                    "pure garbage",
                    "insulted",
                    "totally disappointing",
                    "the worst movie i have ever seen",
                    "disappointed",
                    "rubbish",
                    "unfortunately do not watch really bad",
                    "horrible movie",
                    "crud",
                    "supposed to be bad; actually excels at it",
					
                    "you get what you expect",
                    "pretty average stuff",
                    "slow burner not scary",
                    "tulip mess",
                    "a predictable perfunctory retelling of a renowned tale",
                    "brolin is amazing the film is not",
                    "disaster",
                    "very average",
                    "dares you to watch but you may bail",
                    "not great not horrible",
                    "nicely shot well acted and utterly pointless",
					
                    "scariest movie ever",
                    "i love this movie  this is not the stallone dredd at all ",
                    "simply a wonder and a part of my life now",
                    "simple painful outstandingly beautiful",
                    "gorgeous picture",
                    "deeply moving arousing heartbreaking",
                    "love it",
                    "Great movie",
                    "fantastic movie",
                    "simply brilliant",
                    "incredible personality treatise of a complex individual"])
y_train_text = [["1"],["1"],["1"],["1"],["1"],["1"],["1"],["1"],["1"],["1"],["1"],["5"],["5"],["5"],["5"],["5"],["5"],["5"],["5"],["5"],["5"],["5"],["9"],["9"],["9"],["9"],["9"],["9"],["9"],["9"],["9"],["9"],["9"]]

X_test = np.array(['terrible horrible no good very bad movie pure garbage',
                   'the best worst movie ever',
                   'unfortunately really bad',
				   
                   'so disappointed horrible mess disaster',
                   'perfectly average pointless',
                   'mediocre average predictable',
				   
                   'a stunning arousing portrait of fantastic and simple drama',
                   'very enjoyable love it',
                   'love',
                   'great fantastic movie',
                   'fantastic and simply great'])

				   
y_test_text = [["1"],["1"],["1"],["5"],["5"],["5"],["9"],["9"],["9"],["9"],["9"]]
#target_names = ['1', '5', '9']

lb = preprocessing.MultiLabelBinarizer()
Y = lb.fit_transform(y_train_text)
Y_test = lb.fit_transform(y_test_text)

print ("real dataset:")
print (Y_test)

classifier = Pipeline([
('vectorizer', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
print ("predicted dataset:")
print (predicted)

print ("Accuracy Score: ",accuracy_score(Y_test, predicted))


#classifier = Pipeline([
#    ('vectorizer', CountVectorizer(min_df=1,max_df=2)),
#    ('tfidf', TfidfTransformer()),
#    ('clf', OneVsRestClassifier(LinearSVC()))])
#classifier.fit(X_train, y_train)
#predicted = classifier.predict(X_test)
#for item, labels in zip(X_test, predicted) :
#    print ('%s ===> %s' % (item, ', '.join(target_names[x] for x in labels)))