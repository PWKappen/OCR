import batchGenerator
import utility
import model
import tensorflow as tf
import numpy as np
import os
import completeModel
from scipy import misc

LOCATION_META = 'metaFile.dat'
LOCATION_TRANSLATOR = 'D:/MLProjects/OCR2/translations/encoder_decoder_model_1_translator.txt'
LOCATION_TEST_DATA = 'D:/MLProjects/OCR2/testData.dat'
LOATION_TRAIN_DATA = 'D:/MLProjects/OCR2/trainData.dat'
LOCATION_VALIDATION_DATA = 'D:/MLProjects/OCR2/validData.dat'

#saves tensorflow checkpoint
def save_ckpt(sess, cur_saver, file_path, saved_step):
    direcotry = os.path.dirname(file_path)

    try:
        os.stat(direcotry)
    except:
        os.mkdir(direcotry)

    save_path = cur_saver.save(sess, file_path, global_step=saved_step)
    print("Saved Model to: " + save_path)

#translate characters to numbers
def translate(file_name):
    with open(file_name, 'r') as file:
        translator = dict()
        for line in file:
            split = line.split('\t')
            idx = int(split[0])
            ch = split[1][0]
            if ch in translator.values():
                print(ch + 'gets multiple times translated to')
            translator[idx] = ch
        return translator


batch_size = 20



with tf.device("/cpu:0"):
    # batchGen = batchGenerator.CustomAsynchBatchLoaderBucket('metaFile.dat', 'D:/MLProjects/OCR2/trainData.dat', batch_size, 200, 120, 4)
    # images_batch, labels_batch, seq_len_batch = batchGen.get_inputs()
    #
    # validGen = batchGenerator.CustomAsynchBatchLoaderBucket('metaFile.dat', 'D:/MLProjects/OCR2/validData.dat', batch_size,
    #                                                   30, 5, 4)
    #
    # images_valid, labels_valid, seq_len_valid = validGen.get_inputs()
    #
    # validGen2 = batchGenerator.CustomAsynchBatchLoaderBucketFull('metaFile.dat', 'D:/MLProjects/OCR2/validData.dat',
    #                                                         batch_size,
    #                                                         50, 5, 4)
    #
    # images_valid2, labels_valid2, seq_len_valid2 = validGen2.get_inputs()



    # batchGen = batchGenerator.CustomAsynchBatchLoaderBucketFull('metaFile.dat', 'D:/MLProjects/OCR2/testData.dat',
    #                                                         batch_size,
    #                                                         50, 5, 4)
    #
    # images_batch, labels_batch, seq_len_batch = batchGen.get_inputs()
    #
    # validGen = batchGenerator.CustomAsynchBatchLoaderBucketFull('metaFile.dat', 'D:/MLProjects/OCR2/trainData.dat', batch_size, 200, 120, 4)
    # images_valid, labels_valid, seq_len_valid = validGen.get_inputs()


    translation = translate(LOCATION_TRANSLATOR)

    batchGen = batchGenerator.CustomAsynchBatchLoaderFull(LOCATION_META, LOCATION_TEST_DATA, batch_size,
                                                      50, 30)
    images_batch, labels_batch, seq_len_batch = batchGen.get_inputs()

    batchGen.set_translation(translation)

    validGen = batchGenerator.CustomAsynchBatchLoaderFull(LOCATION_META, LOATION_TRAIN_DATA, batch_size,
                                                      20, 10)
    images_valid, labels_valid, seq_len_valid = validGen.get_inputs()
    
    validGen2 = batchGenerator.CustomAsynchBatchLoaderFull(LOCATION_META, LOCATION_VALIDATION_DATA, batch_size,
                                                      100, 10)
    images_valid2, labels_valid2, seq_len_valid2 = validGen2.get_inputs()


    batchGen.get_max_height()
    maxTimeSteps = batchGen.get_max_seq_len()
    num_cells = 256
    num_classes = batchGen.get_num_chars()
    input_embedding_size = 64
    num_layers = 3
    window_width = 24
    window_offset = 10

with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=4)) as sess:
    # ctc = completeModel.CTCSlidingWindowAttention(seq_len_batch, batch_size, num_cells, num_classes,
    #                                       batchGen.maxWidth, batchGen.maxHeight, window_width, window_offset, num_layers)



    ctc = completeModel.CTCSlidingWindow(seq_len_batch, batch_size, num_cells, num_classes,
                                                  batchGen.maxWidth, batchGen.maxHeight,
                                                  num_layers)



    # ctc = completeModel.CTCSlidingWindowAttention(seq_len_batch, batch_size, num_cells, num_classes,
    #                                      batchGen.maxWidth, batchGen.maxHeight, window_width, window_offset, num_layers)
    #
    #
    #
    #
    ctc_cost_builder = model.ctc_cost_builder()
    rnn_outputs, keep_prob, time_steps_train, extracted_patches, attentions = ctc.build_model(images_batch, batchGen.maxHeight, 3)
    loss, cost, results = ctc_cost_builder(rnn_outputs, labels_batch, seq_len_batch, num_classes, num_cells,
                                           time_steps_train)
    
    rnn_outputs_valid, keep_prob_valid, time_steps_valid, _, _ = ctc.build_model(images_valid, batchGen.maxHeight, 3)
    loss_valid, cost_valid, results_valid = ctc_cost_builder(rnn_outputs_valid, labels_valid, seq_len_valid, num_classes, num_cells,
                                            time_steps_valid)
    
    result_inv_conv, x_placeholder, size_placeholder = model.inv_conv()

    rnn_outputs_valid2, keep_prob_valid2, time_steps_valid2 = ctc.build_model(images_valid2, batchGen.maxHeight, 3)
    loss_valid2, cost_valid2, results_valid2 = ctc_cost_builder(rnn_outputs_valid2, labels_valid2, seq_len_valid2,
                                                             num_classes, num_cells,
                                                             time_steps_valid2)

#     ctc = completeModel.EncoderDecoder(batch_size, num_cells, num_classes,
#                                            batchGen.maxWidth, batchGen.maxHeight)

    oneHot_input = tf.one_hot(labels_batch, num_classes, on_value=1.0, off_value=0.0, dtype=tf.float32)
    results, cost, acc = ctc.build_model(images_batch,seq_len_batch, batchGen.maxHeight, batchGen.maxWidth, True,oneHot_input)

    oneHot_input_valid = tf.one_hot(labels_valid, num_classes, on_value=1.0, off_value=0.0,
                                                    dtype=tf.float32)
    results_valid, cost_valid, acc_valid = ctc.build_model(images_valid, seq_len_valid, batchGen.maxHeight, batchGen.maxWidth, True,
                                         oneHot_input_valid)
    one_hot_input_valid2 = tf.one_hot(labels_valid2, num_classes, on_value=1.0,
                                                                      off_value=0.0,
                                                                      dtype=tf.float32)
    results_valid2, cost_valid2, acc_valid2 = ctc.build_model(images_valid2, seq_len_valid2, batchGen.maxHeight,
                                                           batchGen.maxWidth, False,
                                                           one_hot_input_valid2)
    
    writer = tf.summary.FileWriter(ctc.get_model_name(), sess.graph)
    train_step = tf.train.AdamOptimizer(10e-4).minimize(cost)
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)

#     ckpt = tf.train.get_checkpoint_state("D:\MLProjects\OCR2\ckpt_CTC_encoder_run_2")
#     if ckpt and ckpt.model_checkpoint_path:
#         saver.restore(sess, "D:\MLProjects\OCR2\ckpt_CTC_encoder_run_2\ckpt-52000")
#         print('Load successful')
#     else:
#         print("ERROR loading checkpoint")

    init = tf.global_variables_initializer()
    sess.run(init)

    tf.train.start_queue_runners(sess=sess)
    batchGen.start_threads(sess=sess, n_threads=1)
    validGen.start_threads(sess=sess, n_threads=1)
    validGen2.start_threads(sess=sess, n_threads=1)
    
    ce_summary = tf.summary.scalar('cross entropy', cost)
    ce_summary2 = tf.summary.scalar('cross entropy valid', cost_valid)
    
    cer_placeholder = tf.placeholder(dtype=tf.float32, shape=[])
    cer_summary = tf.summary.scalar('cer', cer_placeholder)
    
    cer_placeholder_full = tf.placeholder(dtype=tf.float32, shape=[])
    cer_summary_full = tf.summary.scalar('cer_full', cer_placeholder_full)
    
    for i in range(1500001):
        _, co, res, lb, ce_sum = sess.run([train_step, cost, results, labels_batch, ce_summary])
        if i % 10 == 0:
            writer.add_summary(ce_sum, i)
            print("Step %d, Cross Entropy: %g" % (i, co))
            if i % 100 == 0:
                co, res, lb = sess.run([cost_valid, results_valid, labels_valid])
    
                wer = list()
    
                for k_idx, k in enumerate(res):
                    correct_string = validGen.decode_Num(lb[k_idx])
                    print(correct_string)
                    decoded = validGen2.decode_Num(k, lb[k_idx])
                    # real_string = ''
                    # for j in decoded:
                    #     real_string += j
                    #     if j == chr(3):
                    #         break
                    print(decoded)
                    wer.append(utility.wer(correct_string, decoded))
                average_wer = np.average(wer)
                print("Step %d, Cross Entropy: %g" % (i, co))
                print(average_wer)
    
                #merged = tf.summary.merge_all()
    
                c1, c2, cer1 = sess.run([ce_summary, ce_summary2, cer_summary], feed_dict={cer_placeholder:average_wer})
                writer.add_summary(c1, i)
                writer.add_summary(c2, i)
                writer.add_summary(cer1, i)
    
                if i % 2000 == 0:
                     file_path = "D:\MLProjects\OCR2\ckpt_" + ctc.get_model_name() + '\ckpt'
                     save_ckpt(sess, saver, file_path, i)
                if i % 1000 == 0:
                    wer = list()
                    while not validGen2.end(sess):
                        res, lb = sess.run([results_valid2, labels_valid2])
                        for k_idx, k in enumerate(res):
                            correct_string = batchGen.decode_Num(lb[k_idx])
                            decoded = validGen2.decode_Num(k, lb[k_idx])
                            wer.append(utility.wer(correct_string, decoded))
    
                    average_wer = np.average(wer)
                    print("Step %d, Average CER validation set: %g" % (i, average_wer))
                    c5 = sess.run(cer_summary_full, feed_dict={cer_placeholder_full: average_wer})
                    writer.add_summary(c5, i)
                    validGen2.reset()

    def word_distribution(fileName, folder, cur_sess, labels, result, loader):
        dictionary = dict()
        dictionary2 = dict()
        i = 0
        while not loader.end(sess):
            lb, res = cur_sess.run([labels, result])
            for idx, vec in enumerate(lb):
                real = loader.decode_label(vec)
                if real in dictionary:
                    dictionary[real] += 1
                else:
                    dictionary[real] = 1
                    if len(real) in dictionary2:
                        dictionary2[len(real)] += 1
                    else:
                        dictionary2[len(real)] = 1
            i +=1
            if i %100 ==0:
                print(i)
        print("start writing to File")
        with open(os.path.join(folder, fileName), 'w') as file:
            for i, j in dictionary2.items():
                file.write(str(i) + '\t')
                file.write(str(j) + '\n')

            for i, j in dictionary.items():
                file.write(i + '\t')
                file.write(str(j) + '\n')

    def calculate_encoding(cur_sess, results2, labels2, keep_prob2, loader, fileName, folder, translationFileName):
        done = False
        dictionary = dict()
        dictionary2 = dict()
        counter = 0
        while not loader.end(sess):
            counter += 1
            res, lb = cur_sess.run([results2, labels2], feed_dict={keep_prob2: 1.0})
            for idx, vec in enumerate(res[0][0]):
                for idx2, char in enumerate(vec):
                    if idx2 < len(lb[idx]):
                        if char != -1:
                            if char in dictionary:
                                dicttmp = dictionary[char]

                                if lb[idx, idx2] in dicttmp:
                                    dicttmp[lb[idx, idx2]] += 1
                                else:
                                    dicttmp[lb[idx, idx2]] = 1
                                dictionary[char] = dicttmp
                                dictionary2[char] += 1
                            else:
                                dictionary[char] = {lb[idx, idx2]: 1}
                                dictionary2[char] = 1
            if counter % 1000 == 0:
                print("here")
        with open(os.path.join(folder, fileName), 'w') as file:
            transFile = open(os.path.join(folder,translationFileName), 'w')
            for keys, values in dictionary.items():
                max = ''
                maxNum = 0
                file.write(str(dictionary2[keys])+'\n')
                for k, v in values.items():
                    if(k != -1):
                        file.write(str(keys) + '\t')
                        file.write(loader.charTransInv[k] + '\t')
                        file.write(str(v) + '\t')
                        file.write(str(v/dictionary2[keys]) + '\n')
                        if v > maxNum:
                            maxNum = v
                            max = loader.charTransInv[k]
                file.write('\n')
                transFile.write(str(keys) + '\t')
                transFile.write(max + '\n')

    def calculate_encoding_Encoder(cur_sess, results2, labels2, loader, fileName, folder, translationFileName):
        done = False
        dictionary = dict()
        dictionary2 = dict()
        counter = 0
        while not loader.end(sess):
            counter += 1
            res, lb = cur_sess.run([results2, labels2])
            for idx, vec in enumerate(res):
                for idx2, char in enumerate(vec):
                    char = np.argmax(char)
                    if idx2 < len(lb[idx]):
                        if char != -1:
                            if char in dictionary:
                                dicttmp = dictionary[char]

                                if lb[idx, idx2] in dicttmp:
                                    dicttmp[lb[idx, idx2]] += 1
                                else:
                                    dicttmp[lb[idx, idx2]] = 1
                                dictionary[char] = dicttmp
                                dictionary2[char] += 1
                            else:
                                dictionary[char] = {lb[idx, idx2]: 1}
                                dictionary2[char] = 1
            if counter % 1000 == 0:
                print("here")
        with open(os.path.join(folder, fileName), 'w') as file:
            transFile = open(os.path.join(folder,translationFileName), 'w')
            for keys, values in dictionary.items():
                max = ''
                maxNum = 0
                file.write(str(dictionary2[keys])+'\n')
                for k, v in values.items():
                    if(k != -1):
                        file.write(str(keys) + '\t')
                        file.write(loader.charTransInv[k] + '\t')
                        file.write(str(v) + '\t')
                        file.write(str(v/dictionary2[keys]) + '\n')
                        if v > maxNum:
                            maxNum = v
                            max = loader.charTransInv[k]
                file.write('\n')
                transFile.write(str(keys) + '\t')
                transFile.write(max + '\n')

    def run_test_window(loader, write_image_precision, results, labels, inputs, windows, keep_prob, cur_sess, error_file_name, folder_name, translation, time_step):
        with open(os.path.join(folder_name, error_file_name), 'w') as error_file:
            wer = list()
            error_file.write('Main image name' + '\t')
            error_file.write('CER'+ '\t')
            error_file.write('Correct Label' + '\t')
            error_file.write('Result Label' + '\t')
            error_file.write('Propability' + '\n')
            lengthDict = dict()
            errorDict = dict()
            done = False
            while not loader.end(sess) and not done:
                res, lb, inp, wind = cur_sess.run([results, labels, inputs, windows], feed_dict={keep_prob: 1.0})
                for k_idx, k in enumerate(res[0][0]):
                    correct_string = loader.decode_Num(lb[k_idx])
                    length = len(correct_string)
                    if length in lengthDict:
                        lengthDict[length] += 1
                    else:
                        lengthDict[length] = 1
                        errorDict[length] = 0
                    decoded = loader.decode_Num_Trans(k, translation)

                    wer.append(utility.wer(correct_string, decoded))
                    errorDict[length] += wer[-1]

                    if wer[-1] > write_image_precision:
                        misc.imsave(os.path.join(folder_name, str(len(wer)) + '.png'), inp[k_idx])
                        for w_idx, w in enumerate(wind[k_idx]):
                            misc.imsave(os.path.join(folder_name, str(len(wer)) + '_' + str(w_idx) + '.png'), w)
                        error_file.write(str(len(wer)) + '.png' + '\t')
                        error_file.write(str(wer[-1]) + '\t')
                        error_file.write(correct_string + '\t')
                        error_file.write(decoded + '\t')
                        error_file.write(str(np.exp(-res[1][k_idx])) + '\n')
            average_wer = np.average(wer)
            print("Average CER test set: %g" % average_wer)
            error_file.write('average cer\t')
            error_file.write('num elements\n')
            error_file.write(str(average_wer) + '\t')
            error_file.write(str(len(wer)) + '\n')
            error_file.write(str(time_step) + '\n')
            for i, k in lengthDict.items():
                error_file.write(str(i) + '\t')
                error_file.write(str(errorDict[i]/k) + '\t')
                error_file.write(str(k) + '\n')

    def run_test_attention(loader, write_image_precision, results, labels, inputs, windows, attentions, keep_prob, result_inv_conv, x_placeholder, size_placeholder, cur_sess, error_file_name, folder_name, translation, time_step):
        with open(os.path.join(folder_name, error_file_name), 'w') as error_file:
            wer = list()
            error_file.write('Main image name' + '\t')
            error_file.write('CER'+ '\t')
            error_file.write('Correct Label' + '\t')
            error_file.write('Result Label' + '\t')
            error_file.write('Propability' + '\n')
            lengthDict = dict()
            errorDict = dict()
            done = False
            while not loader.end(sess) and not done:
                res, lb, inp, wind, att = cur_sess.run([results, labels, inputs, windows, attentions], feed_dict={keep_prob: 1.0})
                for k_idx, k in enumerate(res[0][0]):
                    correct_string = loader.decode_Num(lb[k_idx])
                    length = len(correct_string)
                    if length in lengthDict:
                        lengthDict[length] += 1
                    else:
                        lengthDict[length] = 1
                        errorDict[length] = 0
                    decoded = loader.decode_Num_Trans(k, translation)

                    wer.append(utility.wer(correct_string, decoded))
                    errorDict[length] += wer[-1]

                    if wer[-1] > write_image_precision:
                        misc.imsave(os.path.join(folder_name, str(len(wer)) + '.png'), inp[k_idx])
                        for w_idx, w in enumerate(wind[k_idx]):
                            misc.imsave(os.path.join(folder_name, str(len(wer)) + '_' + str(w_idx) + '.png'), w)
                        for w_idx, w in enumerate(att[k_idx]):
                            if w_idx != (len(att[k_idx]) -1):
                                x_resh = np.reshape(w, (1, 31, 3, 1))
                                att_image = sess.run((result_inv_conv),
                                                     feed_dict={x_placeholder: x_resh, size_placeholder: [31, 3]})
                                new_image = np.resize(np.reshape(att_image, (248, 24)), (241, 24))
                                misc.imsave(
                                    os.path.join(folder_name, str(len(wer)) + '_att_' + '_' + str(w_idx) + '.png'),
                                    new_image)
                                misc.imsave(os.path.join(folder_name,
                                                         str(len(wer)) + '_att_post_conv' + '_' + str(w_idx) + '.png'),
                                            np.reshape(att[k_idx, w_idx], (31, 3)))
                                new_image = new_image * wind[k_idx, w_idx]
                                misc.imsave(
                                    os.path.join(folder_name,
                                                 str(len(wer)) + '_att_combined' + '_' + str(w_idx) + '.png'),
                                    new_image)

                        error_file.write(str(len(wer)) + '.png' + '\t')
                        error_file.write(str(wer[-1]) + '\t')
                        error_file.write(correct_string + '\t')
                        error_file.write(decoded + '\t')
                        error_file.write(str(np.exp(-res[1][k_idx])) + '\n')
            average_wer = np.average(wer)
            print("Average CER test set: %g" % average_wer)
            error_file.write('average cer\t')
            error_file.write('num elements\n')
            error_file.write(str(average_wer) + '\t')
            error_file.write(str(len(wer)) + '\n')
            error_file.write(str(time_step) + '\n')
            for i, k in lengthDict.items():
                error_file.write(str(i) + '\t')
                error_file.write(str(errorDict[i]/k) + '\t')
                error_file.write(str(k) + '\n')


    def run_test_Encode(loader, write_image_precision, results, labels, inputs, cur_sess,
                        error_file_name, folder_name, translation, time_step):
        with open(os.path.join(folder_name, error_file_name), 'w') as error_file:
            wer = list()
            error_file.write('Main image name' + '\t')
            error_file.write('CER' + '\t')
            error_file.write('Correct Label' + '\t')
            error_file.write('Result Label' + '\n')
            lengthDict = dict()
            errorDict = dict()
            done = False
            while not loader.end(sess) and not done:
                res, lb, inp = cur_sess.run([results, labels, inputs])
                for k_idx, k in enumerate(res):

                    correct_string = loader.decode_label(lb[k_idx])
                    length = len(correct_string)
                    if length in lengthDict:
                        lengthDict[length] += 1
                    else:
                        lengthDict[length] = 1
                        errorDict[length] = 0
                    k = np.argmax(k, axis=-1)

                    decoded = loader.decode_Num_Trans(k, translation)
                    wer.append(utility.wer(correct_string, decoded))
                    errorDict[length] += wer[-1]

                    if wer[-1] > write_image_precision:
                        #misc.imsave(os.path.join(folder_name, str(len(wer)) + '.png'), inp[k_idx])
                        error_file.write(str(len(wer)) + '.png' + '\t')
                        error_file.write(str(wer[-1]) + '\t')
                        error_file.write(correct_string + '\t')
                        error_file.write(decoded + '\n')
            average_wer = np.average(wer)
            print("Average CER test set: %g" % average_wer)
            error_file.write('average cer\t')
            error_file.write('num elements\n')
            error_file.write(str(average_wer) + '\t')
            error_file.write(str(len(wer)) + '\n')
            error_file.write(str(time_step) + '\n')
            for i, k in lengthDict.items():
                error_file.write(str(i) + '\t')
                error_file.write(str(errorDict[i] / k) + '\t')
                error_file.write(str(k) + '\n')


    #calculate_encoding_Encoder(sess, results_valid, labels_valid, validGen,'encoder_decoder_model_1_trans.txt','D:/MLProjects/OCR2/translations', 'encoder_decoder_model_1_translator.txt')
    #translation = translate('D:/MLProjects/OCR2/translations/encoder_decoder_model_1_translator.txt')
    #run_test_Encode(batchGen, 0, results, labels_batch, images_batch, sess, 'test_results.txt', 'D:/MLProjects/OCR2/encoder_decoder_model3_results', translation, 52000)
    #word_distribution('distribution_train.txt', 'D:/MLProjects/OCR2/encoder_decoder_model2_results', sess, labels_valid, results_valid, validGen)
    #word_distribution('distribution_test.txt', 'D:/MLProjects/OCR2/encoder_decoder_model2_results', sess, labels_batch, results, batchGen)
