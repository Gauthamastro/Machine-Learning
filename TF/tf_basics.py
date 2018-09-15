def basics():
    import tensorflow as tf
    x = tf.Variable(3,name ="3")
    y = tf.Variable(4,name='4')

    f = x*x*y + y +2

    init = tf.global_variables_initializer()  #prepare a init node

    with tf.Session() as sess:          #initializes all the variables and runs it
        init.run()
        result = f.eval()
        print(result)
def note():
    import tensorflow as tf
    w = tf.constant(3)
    x = w+2
    y = x+5
    z = x*3
    with tf.Session() as sess:
        print(y.eval()) #Note: it computes the value of w and x twice
        print(z.eval()) # during evalutions.. its makes the code lag..

def graphs():
    x1 = tf.Variable(1)
    #This is implemented in the default graph
    # but if we want to create a new graph then
    graph = tf.Graph()
    with graph.as_default():
        x2 = tf.Variable(2)
        #Code for the 2nd graphs....

# Different sessions of same graph doesnt share any state nor variables
# ie each session has its own copy of variables...

'''
using autodiff

gradients = tf.gradients(mse,[theta])[0]
 where mse = the operation or formula on which gradients must be found
 and with respect to list of variables.. here theta only
 it returns a vector of gradients.'''

''' using a placeholder

a placeholder is a node in the graph which outputs data at
runtime as per requirements'''



def placehold():
    import tensorflow as tf
    A = tf.placeholder(tf.float32,shape=(None,3))
    B = A +5
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        B_1 = B.eval(feed_dict={A:[[1,2,3]]})
        B_2 = B.eval(feed_dict={A:[[1,2,3], [4,5,6]]})
    print(B_1)
    print(B_2)
placehold()


'''
Another example is
x_batch,y_batch = our func to fetch the data 
sess.run(training_op,feed_dict={X:x_batch,Y:y_batch})
 this is within with loop for session...
'''

def saving_restoring():
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0 :
                save_path = saver.save(sess,"/tmp/my_model.ckpt")
            sess.run(training_op) # Again training operations


        save_path = saver.save(sess,"/tmp/my_final_model.ckpt")


    with tf.Session() as sess:
        saver.restore(sess,"/tmp/my_model.ckpt")


    ''' the difference is that we call save after the execution part
but the restore in the init part of with session before the training.'''
