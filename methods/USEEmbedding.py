# -*- coding: utf-8 -*-

import tensorflow_hub as hub
import tensorflow as tf
import os

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, '../tmp/TFHub_Module')


class UseEmbed(object):

    def __init__(self, sentences):
        print(tf.test.is_gpu_available())
        print(tf.test.gpu_device_name())
        self.sentences = sentences
        self.embed = hub.Module(filename)
        self.session = tf.Session()
        self.session.run([tf.global_variables_initializer(), tf.tables_initializer()])

    def embed_sentences(self):
        message_embeddings = self.session.run(self.embed(self.sentences))
        return message_embeddings
