import model
import math
import tensorflow as tf

class EncoderDecoder(object):
    def __init__(self, batch_size, num_cells, num_classes, max_width, max_height):
        self.num_cells = num_cells
        self.num_classes = num_classes
        self.max_width = max_width
        self.max_height = max_height
        self.batch_size = batch_size

        self.wc1 = ''
        self.bc1 = ''
        self.wc2 = ''
        self.bc2 = ''
        self.wc3 = ''
        self.bc3 = ''

        self.defined = False
        self.model_template = None

    def build_model_template(self, image_batch, seq_len, max_height, max_width, train, y_oneHot):
        shape = tf.shape(image_batch)
        img_reshaped = tf.reshape(image_batch, [-1, shape[1], shape[2], 1])

        img_features, self.wc3, self.bc3, self.wc2, self.bc2, self.wc1, self.bc1 = model.image_features_org_shape(img_reshaped)

        size_width = max_width
        for i in range(3):
            size_width = math.ceil(size_width / 2)

        size_height = max_height
        for i in range(3):
            size_height = math.ceil(size_height / 2)

        img_features = tf.reshape(img_features, [-1, size_width*size_height*128])
        if train:
            zeros = tf.zeros([self.batch_size, 1, self.num_classes], dtype=tf.float32)
            shape = tf.shape(y_oneHot)
            y_resize = tf.concat([zeros, y_oneHot], 1)
            inputs = tf.slice(y_resize, [0, 0, 0], [self.batch_size, shape[1], self.num_classes])
            inputs = tf.reshape(inputs, [self.batch_size, shape[1], self.num_classes])
            softmax = model.dyn_recurrent_net(self.num_cells, img_features, inputs, seq_len, size_width*size_height * 128, self.batch_size, self.num_classes)
        else:
            softmax = model.dyn_recurrent_net_test(self.num_cells, img_features, seq_len,
                                              size_width*size_height * 128, self.batch_size, self.num_classes, self.num_classes)
        cross_entropy, acc = model.cost(y_oneHot, softmax)
        return softmax, cross_entropy, acc

    def build_model(self, image_batch, seq_len, max_height, max_width, train, y_oneHot):
        if self.defined == False:
            self.model_template = tf.make_template('ctc_model', self.build_model_template)
            self.defined = True
        return self.model_template(image_batch, seq_len, max_height, max_width, train, y_oneHot)

    def get_model_name(self):
        return 'CTC_encoder_run_2'

    def init_conv(self, sess, folder_name, checkpoint_name):
        saver = tf.train.Saver({"Wc1": self.wc1, "bc1": self.bc1, "Wc2": self.wc2, "bc2": self.bc2, "Wc3": self.wc3, "bc3": self.bc3})

        ckpt = tf.train.get_checkpoint_state(folder_name)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, checkpoint_name)
        else:
            print("ERROR loading checkpoint")

class CTCSlidingWindow(object):
    def __init__(self, seq_len_batch, batch_size, num_cells, num_classes, max_width, max_height, num_layers):
        self.seq_len_batch = seq_len_batch
        self.num_cells = num_cells
        self.num_classes = num_classes
        self.max_width = max_width
        self.max_height = max_height
        self.batch_size = batch_size
        self.num_layers = num_layers

        self.wc1 = ''
        self.bc1 = ''
        self.wc2 = ''
        self.bc2 = ''
        self.wc3 = ''
        self.bc3 = ''

        self.defined = False
        self.model_template = None

    def build_model_template(self, image_batch, max_height, extracted_width, extracted_offset, num_max_pools):
        shape = tf.shape(image_batch)
        img_reshaped = tf.reshape(image_batch, [-1, shape[1], shape[2], 1])
        extracted_patches = tf.extract_image_patches(images=img_reshaped, ksizes=[1, max_height, extracted_width, 1], strides=[1, max_height, extracted_offset, 1], rates=[1, 1, 1, 1],
                                                     padding='SAME')
        extracted_patches = tf.reshape(extracted_patches, [-1, max_height, extracted_width, 1])
        tmp_patches = tf.reshape(extracted_patches, [self.batch_size, -1, max_height, extracted_width])
        new_shape = tf.shape(extracted_patches)
        time_steps = tf.to_int32(new_shape[0]/shape[0])

        size_width = extracted_width
        for i in range(num_max_pools):
            size_width = math.ceil(size_width/2)

        img_features, self.wc3, self.bc3, self.wc2, self.bc2, self.wc1, self.bc1 = model.image_features_org_shape(extracted_patches)
        img_features = tf.reshape(img_features, [shape[0], time_steps, 31 * size_width * 128])

        rnn_outputs, keep_prob = model.sliding_window_net(self.num_cells, img_features, self.num_layers)
        return rnn_outputs, keep_prob, time_steps, tmp_patches

    def build_model(self, image_batch, maxHeight, extractedWidth, extracted_offset, num_max_pools):
        if self.defined == False:
            self.model_template = tf.make_template('ctc_model', self.build_model_template)
            self.defined = True
        return self.model_template(image_batch, maxHeight, extractedWidth, extracted_offset, num_max_pools)

    def get_model_name(self):
        return 'CTC_model_run_7'

    def init_conv(self, sess, folder_name, checkpoint_name):
        saver = tf.train.Saver({"Wc1": self.wc1, "bc1": self.bc1, "Wc2": self.wc2, "bc2": self.bc2, "Wc3": self.wc3, "bc3": self.bc3})

        ckpt = tf.train.get_checkpoint_state(folder_name)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, checkpoint_name)
        else:
            print("ERROR loading checkpoint")

class CTCSlidingWindowAttention(object):
    def __init__(self, seq_len_batch, batch_size, num_cells, num_classes, max_width, max_height, extracted_width, width_offset, num_layers):
        self.seq_len_batch = seq_len_batch
        self.num_cells = num_cells
        self.num_classes = num_classes
        self.max_width = max_width
        self.max_height = max_height
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.extracted_width = extracted_width
        self.width_offset = width_offset

        self.wc1 = ''
        self.bc1 = ''
        self.wc2 = ''
        self.bc2 = ''
        self.wc3 = ''
        self.bc3 = ''

        self.defined = False
        self.model_template = None

    def build_model_template(self, image_batch, max_height, num_max_pools):
        shape = tf.shape(image_batch)
        img_reshaped = tf.reshape(image_batch, [-1, shape[1], shape[2], 1])
        extracted_patches = tf.extract_image_patches(images=img_reshaped, ksizes=[1, max_height, self.extracted_width, 1],
                                                     strides=[1, max_height, self.width_offset, 1], rates=[1, 1, 1, 1],
                                                     padding='VALID')
        extracted_patches = tf.reshape(extracted_patches, [-1, max_height, self.extracted_width, 1])
        tmp_patches = tf.reshape(extracted_patches, [self.batch_size, -1, max_height, self.extracted_width])
        new_shape = tf.shape(extracted_patches)
        time_steps = tf.to_int32(new_shape[0]/shape[0])

        size_width = self.extracted_width
        for i in range(num_max_pools):
            size_width = math.ceil(size_width/2)

        print(size_width)
        img_features, self.wc3, self.bc3, self.wc2, self.bc2, self.wc1, self.bc1 = model.image_features_org_shape(extracted_patches)
        img_features = tf.reshape(img_features, [shape[0], time_steps, 31 * size_width, 128])
        img_features = tf.transpose(img_features, perm=[1,0,3,2])
        rnn_outputs, keep_prob, attentions = model.sliding_window_net_attention_init(self.num_cells, img_features, self.num_layers, self.batch_size, size_width * 31 * 128, size_width * 31, time_steps)
        return rnn_outputs, keep_prob, time_steps, tmp_patches, tf.transpose(attentions, perm=[1,0,2])

    def build_model(self, image_batch, maxHeight, num_max_pools):
        if self.defined == False:
            self.model_template = tf.make_template('ctc_model', self.build_model_template)
            self.defined = True
        return self.model_template(image_batch, maxHeight, num_max_pools)

    def get_model_name(self):
        return 'CTC_attention_t2_model_run_4'

    def init_conv(self, sess, folder_name, checkpoint_name):
        saver = tf.train.Saver({"Wc1": self.wc1, "bc1": self.bc1, "Wc2": self.wc2, "bc2": self.bc2, "Wc3": self.wc3, "bc3": self.bc3})

        ckpt = tf.train.get_checkpoint_state(folder_name)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, checkpoint_name)
        else:
            print("ERROR loading checkpoint")