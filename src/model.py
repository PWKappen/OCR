import tensorflow as tf

def weight_variable_conv2D(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32))

def weight_variable(shape, name):
    return tf.get_variable(name, shape, initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float32))

def bias_variable(shape, name):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(0.0001, dtype=tf.float32))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def image_features(x, reduction_size):
    with tf.variable_scope('conv') as vs:
        W_conv1 = weight_variable_conv2D([3, 3, 1, 32], 'Wc1')
        b_conv1 = bias_variable([32], 'bc1')

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable_conv2D([3, 3, 32, 64], 'Wc2')
        b_conv2 = bias_variable([64], 'bc2')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_conv3 = weight_variable_conv2D([3, 3, 64, 64], 'Wc3')
        b_conv3 = bias_variable([64], 'bc3')

        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        W_conv4 = weight_variable_conv2D([3, 3, 64, 64], 'Wc4')
        b_conv4 = bias_variable([64], 'bc4')

        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)

        return tf.reshape(h_pool4, [-1, reduction_size])

def image_features(x, reduction_size, num_cells):
    with tf.variable_scope('conv') as vs:
        W_conv1 = weight_variable_conv2D([3, 3, 1, 32], 'Wc1')
        b_conv1 = bias_variable([32], 'bc1')

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable_conv2D([3, 3, 32, 64], 'Wc2')
        b_conv2 = bias_variable([64], 'bc2')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_conv3 = weight_variable_conv2D([3, 3, 64, 64], 'Wc3')
        b_conv3 = bias_variable([64], 'bc3')

        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        W_conv4 = weight_variable_conv2D([3, 3, 64, 64], 'Wc4')
        b_conv4 = bias_variable([64], 'bc4')

        h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
        h_pool4 = max_pool_2x2(h_conv4)

        features = tf.reshape(h_pool4, [-1, reduction_size])

        W_fc1 = weight_variable([reduction_size, num_cells], 'Wf1')
        b_fc1 = bias_variable([num_cells], 'bf1')

        init_features = tf.matmul(features, W_fc1) + b_fc1
        return init_features

def image_features_org_shape(x):
    with tf.variable_scope('conv') as vs:
        W_conv1 = weight_variable_conv2D([5, 5, 1, 48], 'Wc1')
        b_conv1 = bias_variable([48], 'bc1')

        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        W_conv2 = weight_variable_conv2D([5, 5, 48, 64], 'Wc2')
        b_conv2 = bias_variable([64], 'bc2')

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = max_pool_2x2(h_conv2)

        W_conv3 = weight_variable_conv2D([5, 5, 64, 128], 'Wc3')
        b_conv3 = bias_variable([128], 'bc3')

        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        h_pool3 = max_pool_2x2(h_conv3)

        return h_pool3, W_conv3, b_conv3, W_conv2, b_conv2, W_conv1, b_conv1

def inv_conv():
    x = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 1])
    size = tf.placeholder(dtype=tf.int32, shape=[2])
    filter = tf.constant(1, shape=[5, 5, 1, 1], dtype=tf.float32)
    pool3 = tf.image.resize_images(x, size=size*2, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    conv3 = conv2d(pool3, filter)
    pool2 = tf.image.resize_images(conv3, size=size*4, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    conv2 = conv2d(pool2, filter)
    pool1 = tf.image.resize_images(conv2, size=size*8, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    conv1 = conv2d(pool1, filter)

    return conv1, x, size
# def recurrent_net_const(num_cells, batch_size, reduction_size, features, seq_len):
#     cell = tf.contrib.rnn.GRUCell(num_cells)
#
#     def loop_fn(time, cell_output, cell_state, loop_state):
#         emit_output = cell_output
#         if cell_output is None:
#             next_cell_state = cell.zero_state(batch_size, tf.float32)
#         else:
#             next_cell_state = cell_state
#         elements_finished = (time >= seq_len)
#         finished = tf.reduce_all(elements_finished)
#         next_input = tf.cond(
#             finished,
#             lambda: tf.zeros([batch_size, reduction_size], dtype=tf.float32),
#             lambda: features)
#         next_loop_state = None
#         return (elements_finished, next_input, next_cell_state,
#                 emit_output, next_loop_state)
#
#     outputs, final_state, _ = tf.nn.raw_rnn(cell, loop_fn=loop_fn)
#
#     outputs_stack = outputs.pack()
#     outputs_stack = tf.transpose(outputs_stack, perm=[1, 0, 2])
#
#     keep_prob = tf.placeholder(tf.float32)
#     rnn_outputs = tf.nn.dropout(outputs_stack, keep_prob=keep_prob)
#
#     return rnn_outputs, keep_prob

def dyn_recurrent_net_test(num_cells, image, seq_len, unrolled_size, batch_size, input_size, num_classes):
    cell = tf.contrib.rnn.GRUCell(num_cells)
    w_init = weight_variable([unrolled_size, num_cells], name='wf_init')
    b_init = bias_variable([num_cells], name='bf_init')

    eow_char = 69
    max_len = 20
    init_state = tf.nn.relu(tf.matmul(image, w_init) + b_init)

    W_fc1 = weight_variable([num_cells, num_classes], 'wf_1')
    b_fc1 = bias_variable([num_classes], 'bf_1')

    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output
        if cell_output is None:
            next_cell_state = init_state
        else:
            next_cell_state = cell_state
        elements_finished = time >= max_len
        if cell_output is None:
            next_input = tf.zeros([batch_size, input_size])
            next_loop_state = tf.zeros([batch_size], dtype=tf.int64)
        else:
            next_loop_state = tf.argmax(tf.nn.softmax(tf.matmul(cell_output, W_fc1) + b_fc1), 1)
            next_input = tf.cond(
                elements_finished,
                lambda: tf.zeros([batch_size, input_size], dtype=tf.float32),
                lambda: tf.one_hot(next_loop_state,num_classes, dtype=tf.float32))

        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    outputs, final_state, _ = tf.nn.raw_rnn(cell, loop_fn=loop_fn)

    outputs_stack = outputs.stack()
    outputs_stack = tf.transpose(outputs_stack, perm=[1, 0, 2])

    stacked_rnn_output = tf.reshape(outputs_stack, [-1, num_cells])
    softmax = tf.nn.softmax(tf.matmul(stacked_rnn_output, W_fc1) + b_fc1)
    r_softmax = tf.reshape(softmax, [batch_size, -1, num_classes])

    return r_softmax

def dyn_recurrent_net(num_cells, image, inputs, seq_len, unrolled_size, batch_size, num_classes):
    cell = tf.contrib.rnn.GRUCell(num_cells)

    w_init = weight_variable([unrolled_size, num_cells], name='wf_init')
    b_init = bias_variable([num_cells], name='bf_init')

    init_state = tf.nn.relu(tf.matmul(image, w_init) + b_init)

    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, seq_len, init_state)

    stacked_rnn_output = tf.reshape(outputs, [-1, num_cells])
    W_fc1 = weight_variable([num_cells, num_classes], 'wf_1')
    b_fc1 = bias_variable([num_classes], 'bf_1')

    softmax = tf.nn.softmax(tf.matmul(stacked_rnn_output, W_fc1) + b_fc1)
    r_softmax = tf.reshape(softmax, [batch_size, -1, num_classes])

    return r_softmax

def sliding_window_net(num_cells, features, num_layers):

    cell = list()
    for i in range(num_layers):
        cell.append(tf.contrib.rnn.GRUCell(num_cells))

    # extracted_patches = tf.extract_image_patches(images=features, ksizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding='SAME')
    # ext_patches = tf.shape(extracted_patches)
    # extracted_patches = tf.reshape(extracted_patches, [-1, ext_patches[1]*ext_patches[2], num_values])

    stack = tf.contrib.rnn.MultiRNNCell(cell)

    outputs, final_state = tf.nn.dynamic_rnn(stack, features, dtype=tf.float32, swap_memory=True)

    keep_prob = tf.placeholder(tf.float32)
    rnn_outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)

    return rnn_outputs, keep_prob

def sliding_window_net_attention(num_cells, features, num_layers, batch_size, unrolledSize, imageSize, seq_len):
    depth = 128

    cell = tf.contrib.rnn.GRUCell(num_cells)
    cellList = list()
    for i in range(num_layers):
        cellList.append(tf.contrib.rnn.GRUCell(num_cells))

    cell = tf.contrib.rnn.MultiRNNCell(cellList)
    res_features = tf.reshape(features, [-1, unrolledSize])
    w_at = weight_variable([num_cells, imageSize], name='w_at')
    u_at = weight_variable([unrolledSize, imageSize], name='u_at')
    g_at = weight_variable([imageSize, imageSize], name='g_at')
    v_at = weight_variable([imageSize], name='v_at')

    matmul_res = tf.reshape(tf.matmul(res_features, u_at),[-1, batch_size, imageSize])

    attentions = tf.TensorArray(size=seq_len + 1, dtype=tf.float32, clear_after_read=False)

    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output
        if cell_output is None:
            next_cell_state = cell.zero_state(batch_size, tf.float32)
        else:
            next_cell_state = cell_state

        elements_finished = (time >= seq_len)
        finished = tf.reduce_all(elements_finished)
        if cell_output is None:
            next_loop_state = attentions.write(time, tf.nn.softmax(v_at*tf.nn.tanh(
                    tf.matmul(cellList[0].zero_state(batch_size, tf.float32), w_at) + tf.reshape(tf.slice(matmul_res, [time, 0, 0], [1, batch_size, imageSize]),
                                   [batch_size, imageSize]))))


            tmp_state = tf.reshape(tf.tile(next_loop_state.read(time), [1,depth]),[1,batch_size,depth, imageSize])

            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, depth], dtype=tf.float32),
                lambda: tf.reshape(tf.reduce_sum(tf.slice(features, [time, 0,0,0], [1, batch_size, depth, imageSize]) * tmp_state, 3), shape=[batch_size, depth])
            )
        else:
            # attention = tf.nn.softmax(
            #         tf.matmul(tf.concat([tf.reshape(tf.slice(features, [time, 0, 0], [1, batch_size, width * height]),
            #                                         [batch_size, width * height]),
            #
            #                               cell_state], 1), w_at) + b_at)
            next_loop_state = loop_state.write(time, tf.cond(
                finished,
                lambda: tf.zeros([batch_size, imageSize], dtype=tf.float32),
                lambda: tf.nn.softmax(v_at*tf.nn.tanh(tf.matmul(tf.reshape(tf.slice(cell_state, [0, 0, 0],
                                                                                                            [1, batch_size, num_cells]),
                                   [batch_size, num_cells]), w_at) + tf.matmul(loop_state.read(time-1), g_at) + tf.reshape(tf.slice(matmul_res, [time, 0, 0], [1, batch_size, imageSize]),
                                   [batch_size, imageSize])))))
            tmp_state = tf.reshape(tf.tile(next_loop_state.read(time), [1, depth]), [1, batch_size, depth, imageSize])
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, depth], dtype=tf.float32),
                lambda: tf.reshape(tf.reduce_sum(tf.slice(features, [time, 0,0,0], [1, batch_size, depth, imageSize]) * tmp_state, 3), shape=[batch_size, depth])
            )

        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    outputs, final_state, attentions = tf.nn.raw_rnn(cell, loop_fn=loop_fn, swap_memory=True, parallel_iterations=False)
    outputs = outputs.stack()
    outputs = tf.transpose(outputs, perm=[1,0,2])
    keep_prob = tf.placeholder(tf.float32)
    rnn_outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)

    return rnn_outputs, keep_prob, attentions.stack()

def sliding_window_net_attention_init(num_cells, features, num_layers, batch_size, unrolledSize, imageSize, seq_len):

    w_init = weight_variable([unrolledSize, num_cells], name='w_init')
    b_init = bias_variable([num_cells], name='b_init')

    init_vec = tuple([tf.matmul(tf.reshape(tf.slice(features, [0, 0, 0, 0], [1, batch_size, 128, imageSize]),
               [batch_size, unrolledSize]), w_init) + b_init, tf.zeros([batch_size, num_cells], dtype=tf.float32), tf.zeros([batch_size, num_cells], dtype=tf.float32)])



    cellList = list()
    for i in range(num_layers):
        cellList.append(tf.contrib.rnn.GRUCell(num_cells))

    cell = tf.contrib.rnn.MultiRNNCell(cellList)

    res_features = tf.reshape(features, [-1, unrolledSize])
    w_at = weight_variable([num_cells, imageSize], name='w_at')
    u_at = weight_variable([unrolledSize, imageSize], name='u_at')
    g_at = weight_variable([imageSize, imageSize], name='g_at')
    v_at = weight_variable([imageSize], name='v_at')

    matmul_res = tf.reshape(tf.matmul(res_features, u_at),[-1, batch_size, imageSize])

    attentions = tf.TensorArray(size=seq_len+1, dtype=tf.float32, clear_after_read=False)


    def loop_fn(time, cell_output, cell_state, loop_state):
        emit_output = cell_output

        if cell_output is None:
            next_cell_state = init_vec
        else:
            next_cell_state = cell_state
        elements_finished = (time >= seq_len)
        finished = tf.reduce_all(elements_finished)
        if cell_output is None:
            next_loop_state = attentions.write(time, tf.nn.softmax(v_at*tf.nn.tanh(
                    tf.matmul(cellList[0].zero_state(batch_size, tf.float32), w_at) + tf.reshape(tf.slice(matmul_res, [time, 0, 0], [1, batch_size, imageSize]),
                                   [batch_size, imageSize]))))
            tmp_state = tf.reshape(tf.tile(next_loop_state.read(time), [1,128]),[1,batch_size,128, imageSize])

            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, 128], dtype=tf.float32),
                lambda: tf.reshape(tf.reduce_sum(tf.slice(features, [time, 0,0,0], [1, batch_size, 128, imageSize]) * tmp_state, 3), shape=[batch_size, 128])
            )
        else:

            next_loop_state = loop_state.write(time, tf.cond(
                finished,
                lambda: tf.zeros([batch_size, imageSize], dtype=tf.float32),
                lambda: tf.nn.softmax(v_at*tf.nn.tanh(tf.matmul(tf.reshape(tf.slice(cell_state, [0, 0, 0],
                                                                                                            [1, batch_size, num_cells]),
                                   [batch_size, num_cells]), w_at) + tf.matmul(loop_state.read(time-1), g_at) + tf.reshape(tf.slice(matmul_res, [time, 0, 0], [1, batch_size, imageSize]),
                                   [batch_size, imageSize])))))
            tmp_state = tf.reshape(tf.tile(next_loop_state.read(time), [1, 128]), [1, batch_size, 128, imageSize])
            next_input = tf.cond(
                finished,
                lambda: tf.zeros([batch_size, 128], dtype=tf.float32),
                lambda: tf.reshape(tf.reduce_sum(tf.slice(features, [time, 0,0,0], [1, batch_size, 128, imageSize]) * tmp_state, 3), shape=[batch_size, 128])
            )


        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    outputs, final_state, attentions = tf.nn.raw_rnn(cell, loop_fn=loop_fn, swap_memory=True, parallel_iterations=False)
    outputs = outputs.stack()
    outputs = tf.transpose(outputs, perm=[1,0,2])
    keep_prob = tf.placeholder(tf.float32)
    rnn_outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)

    return rnn_outputs, keep_prob, attentions.stack()

def ctc_cost_builder():
    return tf.make_template('ctc_cost', ctc_cost)

def ctc_cost(rnn_outputs, labels, seq_len, num_classes, num_cells, num_steps_test):
    with tf.variable_scope('soft') as vs:
        stacked_rnn_output = tf.reshape(rnn_outputs, [-1, num_cells])
        eff_num_classes = num_classes + 1  # padding output
        W_fc1 = weight_variable([num_cells, eff_num_classes], 'Wf1')
        b_fc1 = bias_variable([eff_num_classes],'bf1')

        pre_soft = tf.matmul(stacked_rnn_output, W_fc1) + b_fc1
        pre_soft = tf.nn.softmax(pre_soft)
        pre_soft = tf.reshape(pre_soft, [-1, num_steps_test, eff_num_classes])

        shape = tf.shape(pre_soft)

        input_length = tf.tile([shape[1]], [shape[0]])
        loss = tf.contrib.keras.backend.ctc_batch_cost(labels, pre_soft, input_length, seq_len)
        results = tf.contrib.keras.backend.ctc_decode(pre_soft, input_length, greedy=False)

        cost = tf.reduce_mean(loss)

        return loss, cost, results

def rnn_softmax(rnn_outputs, num_cells, num_classes, batch_size):
    with tf.variable_scope('soft') as vs:
        stacked_rnn_output = tf.reshape(rnn_outputs, [-1, num_cells])
        W_fc1 = weight_variable([num_cells, num_classes], 'Wf1')
        b_fc1 = bias_variable([num_classes], 'Bf1')

        softmax = tf.nn.softmax(tf.matmul(stacked_rnn_output, W_fc1) + b_fc1)
        r_softmax = tf.reshape(softmax, [batch_size, -1, num_classes])
        return r_softmax

def cost(y_oneHot, softmax):
    cross_entropy = y_oneHot * tf.log(softmax)
    cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
    mask = tf.sign(tf.reduce_max(tf.abs(y_oneHot), reduction_indices=2))

    cross_entropy *= mask

    cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
    cross_entropy /= tf.reduce_sum(mask, reduction_indices=1)
    cross_entropy = tf.reduce_mean(cross_entropy)

    accuracy = tf.to_float(tf.equal(tf.argmax(softmax, 2), tf.argmax(y_oneHot, 2)))
    accuracy *= mask
    accuracy = tf.reduce_sum(accuracy, reduction_indices=1)
    accuracy /= tf.reduce_sum(mask, reduction_indices=1)
    accuracy = tf.reduce_mean(accuracy)

    return cross_entropy, accuracy

# with tf.device("/cpu:0"):
#     batchGen = batchGenerator.CustomAsynchBatchLoader('metaFile.dat', 'D:/MLProjects/OCR2/trainData.dat', batch_size, 200, 20)
#     images_batch, labels_batch, seq_len_batch = batchGen.get_inputs()
#
#     batchGen.get_max_height()
#     maxTimeSteps = batchGen.get_max_seq_len()
#     num_cells = 800
#     reduction_size = 69632
#     num_classes = batchGen.get_num_chars()
#     input_embedding_size = 64



# y_oneHot = tf.one_hot(labels_batch, num_classes, on_value=1.0, off_value=0.0, dtype=tf.float32)
#
# zeros = tf.zeros([batch_size, 1, num_classes], dtype=tf.float32)
# shape = tf.shape(y_oneHot)
# y_resize = tf.concat([zeros, y_oneHot], 1)
# y_resize = tf.slice(y_resize, [0, 0, 0], shape)
#
# #y_resize = tf.reshape(y_oneHot, [-1, num_classes])
# #W_fc1 = model.weight_variable([num_classes, input_embedding_size])
#
# #input_embedding = tf.matmul(y_resize, W_fc1)
#
# #input_embedding = tf.reshape(input_embedding, [batch_size, -1, input_embedding_size])
# #input_embedding = tf.transpose(input_embedding, perm=[1, 0, 2])
#
# y_resize = tf.reshape(y_resize, [-1, num_classes])
# W_fc1 = model.weight_variable([num_classes, input_embedding_size])
#
# input_embedding = tf.matmul(y_resize, W_fc1)
#
# input_embedding = tf.reshape(input_embedding, [batch_size, -1, input_embedding_size])
#
# x = tf.reshape(images_batch, [-1, batchGen.maxHeight, batchGen.maxWidth, 1])
#
# im_features = model.image_features(x, reduction_size, num_cells)
#
# rnn_outputs, keep_prob = model.dyn_recurrent_net(num_cells, im_features, input_embedding, seq_len_batch)
#
# softmax = model.rnn_softmax(rnn_outputs, num_cells, num_classes, batch_size)
#
# cross_entropy, accuracy = model.cost(y_oneHot, softmax)
# perplexity = tf.exp(cross_entropy)
#
# train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
#
# tf.summary.scalar('accuracy', accuracy)
# tf.summary.scalar('cross entropy', cross_entropy)
# tf.summary.scalar('perplexity', perplexity)
# merged = tf.summary.merge_all()
#
# #saver = tf.train.Saver()
#
# with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4)) as sess:
#     writer = tf.summary.FileWriter(model_name, sess.graph)
#
#     #ckpt = tf.train.get_checkpoint_state("ckpt")
#     #if ckpt and ckpt.model_checkpoint_path:
#     #    saver.restore(sess, "ckpt")
#     #else:
#     #    print("ERROR loading checkpoint")
#
#     init = tf.global_variables_initializer()
#     sess.run(init)
#
#     tf.train.start_queue_runners(sess=sess)
#     batchGen.start_threads(sess=sess, n_threads=1)
#
#     for i in range(50000):
#         _, ce, soft, lab, ro, sl, ac, mer, per = sess.run(
#             [train_step, cross_entropy, softmax, labels_batch, rnn_outputs, seq_len_batch, accuracy, merged, perplexity], feed_dict={keep_prob: 0.5})
#         if i % 10 == 0:
#             print("Step %d, Cross Entropy: %g, Accuracy: %g, Perplexity: %g" % (i, ce, ac, per))
#             if i % 500 == 0:
#                 writer.add_summary(mer, i)
#                 print(soft[0, 0])
#                 #print(ro[0, 4])
#                 #print(ro[0, 5])
#                 print(sl[0])
#                 print(lab[0])
#
#                 for z in range(soft.shape[1]):
#                     print(soft[0, z, lab[0, z]])
#                 print(ce)
#                # if i % 500 == 0 and i != 0:
#                     #file_path = "ckpt"
#                     #save_path = saver.save(sess, file_path, global_step=i)
#                     #print("Saved Model to: " + save_path)