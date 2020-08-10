import unittest
import logging
import topic_model as models

TEST_SENTENCE = 'In the time since the industrial revolution the climate has increasingly been affected by human ' \
                'activities that are causing global warming and climate change.'
TEST_CORPUS = './data/20ng.txt'
TEST_LABELS = './data/20ng_labels.txt'


# logging.basicConfig(level=logging.DEBUG)


class MainTest(unittest.TestCase):

    # def test_train(self):
    #     for model in models.__all__:
    #         m = model()
    #         res = m.train(data=TEST_CORPUS, num_topics=20)
    #         self.assertEqual(res, 'success', '[%s] Problems in training.' % model)
    #
    # def test_predict(self):
    #     for model in models.__all__:
    #         m = model()
    #         res = m.predict(TEST_SENTENCE, topn=5)
    #         print(res)
    #         if 'message' in res:
    #             self.assertEqual(res['message'], 'not implemented for this model',
    #                              '[%s] Unexpected output for the prediction')
    #         else:
    #             self.assertIsInstance(res, list, '[%s] Predict output should be a list.' % model)
    #             self.assertEqual(len(res), 5, '[%s] Predict output should match topn.' % model)
    #             self.assertIsInstance(res[0], tuple,
    #                                   '[%s] Predictions should be represented as tuple.' % model)

    # def test_topics(self):
    #     for model in models.__all__:
    #         m = model()
    #
    #         res = m.topics
    #         print(res)
    #         self.assertIsInstance(res, list, '[%s] Topics output should be a list.' % model)
    #         self.assertIn('words', res[0], '[%s] Topics output should be like {words: [], weights: [] }.' % model)
    #
    # def test_given_topic(self):
    #     for model in models.__all__:
    #         m = model()
    #
    #         res = m.topic(0)
    #         print(res)
    #         self.assertIn('words', res, '[%s] Topics output should be like {words: [], weights: [] }.' % model)
    #
    # def test_coherence(self):
    #     for model in models.__all__:
    #         m = model()
    #         m.load()
    #
    #         res = m.coherence(datapath=TEST_CORPUS)
    #         print(res)
    #         self.assertIsInstance(res, dict, '[%s] Coherence output should be a dict.' % model)
    #         self.assertIn('c_v', res,
    #                       '[%s] Coherence output should be like {c_v_per_topics: [], c_v: 0.01, c_v_std: 0.01 }.' % model)
    #         self.assertIsInstance(res['c_v'], float, '[%s] Coherence output should be a floating point.' % model)


    def test_corpus_predictions(self):
        for model in models.__all__:
            m = model()
            m.load()
            print(model)

            res = m.get_corpus_predictions()
            # print(res)
            self.assertIsInstance(res, list, '[%s] Corpus prediction output should be a list of lists.' % model)
            self.assertIsInstance(res[0], list, '[%s] Corpus prediction output should be a list of lists.' % model)
            self.assertIsInstance(res[0][0], tuple,
                                  '[%s] Corpus prediction topics should be represented as tuple.' % model)

    def test_evaluate(self):
        with open(TEST_LABELS, 'r') as f:
            labels = [x.strip() for x in f.readlines()]

        for model in models.__all__:
            m = model()
            m.load()
            print(model)

            res = m.get_corpus_predictions(topn=1)
            v = m.evaluate(res, labels, metric='nmi')
            v2 = m.evaluate(res, labels, metric='purity')
            print(v)

            self.assertIsInstance(v, float, '[%s] Evaluate NMI should return a float.' % model)
            self.assertIsInstance(v2, float, '[%s] Evaluate Purity should return a float.' % model)


if __name__ == '__main__':
    unittest.main()
