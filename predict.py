import os
import cv2
import timeit
import numpy as np
import tensorflow as tf

camera = cv2.VideoCapture(1)

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
               in tf.gfile.GFile('/home/sai/face/katomaran attendence/labels.txt')]

def grabVideoFeed():
    grabbed, frame = camera.read()
    return frame if grabbed else None

def initialSetup():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    start_time = timeit.default_timer()

    # This takes 2-5 seconds to run
    # Unpersists graph from file
    with tf.gfile.FastGFile('/home/sai/face/katomaran attendence/model_graph.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

    print ('Took {} seconds to unpersist the graph'.format(timeit.default_timer() - start_time))

initialSetup()

with tf.Session() as sess:
    start_time = timeit.default_timer()

    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    print( 'Took {} seconds to feed data to graph'.format(timeit.default_timer() - start_time))
    sampleNum= 0
    while True:
        frame = grabVideoFeed()

        if frame is None:
            raise SystemError('Issue grabbing the frame')

        frame = cv2.resize(frame, (299, 299), interpolation=cv2.INTER_CUBIC)

        cv2.imshow('Main', frame)

        # adhere to TS graph input structure
        numpy_frame = np.asarray(frame)
        numpy_frame = cv2.normalize(numpy_frame.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
        numpy_final = np.expand_dims(numpy_frame, axis=0)

        start_time = timeit.default_timer()

        # This takes 2-5 seconds as well
        predictions = sess.run(softmax_tensor, {'Mul:0': numpy_final})

        print ('Took {} seconds to perform prediction'.format(timeit.default_timer() - start_time))

        start_time = timeit.default_timer()

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        print ('Took {} seconds to sort the predictions'.format(timeit.default_timer() - start_time))

       
        human_string = label_lines[top_k[0]]
        score = predictions[0][top_k[0]]
        if (score >= 0.9):
            print('%s (score = %.5f)' % (human_string, score))
            cv2.imwrite("/home/sai/face/katomaran attendence/data/User"+str(sampleNum)+".jpg", frame)

        elif (score <=0.4):
            print("unknown_person")

        print ('********* Session Ended *********')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sess.close()
            break

camera.release()
cv2.destroyAllWindows()
