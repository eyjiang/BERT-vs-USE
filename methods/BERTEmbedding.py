# -*- coding: utf-8 -*-

from bert_serving.client import BertClient


class BertEmbed(object):

    def __init__(self, sentences):
        self.sentences = sentences

    def embed_sentences(self):
        bc = BertClient()
        embedding = bc.encode(self.sentences)
        return embedding
