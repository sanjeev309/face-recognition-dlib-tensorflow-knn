import tensorflow as tf
import numpy as np
import util

def KnnModel(vector=np.zeros([1,128]), infer = False):

    x_ = util.getKNNx()
    x_ = np.vstack(x_)

    y = util.getY()
    y_ = np.eye(len(set(y)))[y]

    train_indices = np.random.choice(len(x_),round(len(x_) * 0.8),replace=False)
    test_indices =np.array(list(set(range(len(x_))) - set(train_indices)))

    x_train = x_[train_indices]
    y_train = y_[train_indices]
    x_test = x_[test_indices]
    y_test = y_[test_indices]

    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)

    k = 6

    X = tf.placeholder(shape=[None,128],dtype=tf.float64)
    Y = tf.placeholder(shape=[None,len(y_[0])],dtype = tf.int8)
    X_test = tf.placeholder(shape=[None,128],dtype=tf.float64)

    # print(X)
    # print(Y)
    # print(X_test)
    # print(x_test.shape)

    EuclideanDistance = tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(X,tf.expand_dims(X_test,1)))),axis=2)

    # print(EuclideanDistance)
    _, top_k_indices = tf.nn.top_k(tf.negative(EuclideanDistance), k=k)
    top_k_label = tf.gather(Y, top_k_indices)

    sum_predictions = tf.reduce_sum(top_k_label, axis=1)
    prediction = tf.argmax(sum_predictions, axis=1)

    sess = tf.Session()
    # print(y_train.shape)
    # print(Y)

    if infer:
        vector = np.expand_dims(vector,axis=0)
        predict = sess.run(prediction, feed_dict={X: x_train,
                                                         X_test: vector,
                                                         Y: y_train})

        print("Person "+str(predict) +" recognized")
    else:
        predict = sess.run(prediction, feed_dict={X: x_train,
                                                             X_test: x_test,
                                                             Y: y_train})

    # evaluation
    accuracy = 0
    for pred, actual in zip(predict, y_test):
        if pred == np.argmax(actual):
            accuracy += 1

    print(accuracy / len(predict))

KnnModel()