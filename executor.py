import rocketqa
import numpy as np
from jina import Executor, requests
from jina.logging.logger import JinaLogger


class RocketQACrossEncoder(Executor):
    """
    rocketQAcrossEncoder
    """

    def __init__(
        self,
        model_name="zh_dureader_ce",
        use_cuda=True,
        device_id=0,
        batch_size=32,
        k=3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)
        self.available_models_list = list(rocketqa.available_models())
        """
        available_models_list = ['v1_marco_de', 'v1_marco_ce', 'v1_nq_de',
                            'v1_nq_ce', 'pair_marco_de', 'pair_nq_de',
                            'v2_marco_de', 'v2_marco_ce', 'v2_nq_de',
                            'zh_dureader_de', 'zh_dureader_ce',
                            'zh_dureader_de_v2', 'zh_dureader_ce_v2']
        """
        if '_ce' not in model_name:
            raise ValueError('need ce model name')
        if model_name not in self.available_models_list:
            raise ValueError(
                f'The ``model_name`` parameter should be in available models list, but got {model_name}'
            )
        self.k = k
        self.gth_k = k
        self.model = rocketqa.load_model(
            model=model_name,
            use_cuda=use_cuda,
            device_id=device_id,
            batch_size=batch_size,
        )

    @requests(on="/search")
    def rerank(self, docs, **kwargs):
        """
        Use the cross model to score and rerank retrieved documents
        :param docs: documents sent to the encoder.
        :param **kwargs: parameter for keyword arguments.
        """
        if docs is not None:
            if len(docs) != 0:
                for doc in docs:
                    if len(doc.matches) != 0:
                        temp_matched = doc.matches
                        str_dict = {}
                        str_list = []
                        if self.gth_k > len(doc.matches):
                            self.k = len(doc.matches)
                        else:
                            self.k = self.gth_k
                        for i, m in enumerate(doc.matches):
                            str_list.append(m.text)
                            str_dict[i] = m.id
                        doc.matches = []
                        scores = list(
                            self.model.matching(
                                query=[doc.text] * len(str_list), para=str_list
                            )
                        )
                        scores = np.array(scores).argsort()
                        for i in range(self.k):
                            doc.matches.append(temp_matched[str_dict[scores[-(1 + i)]]])
