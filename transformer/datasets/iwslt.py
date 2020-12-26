import torch
from torchtext import data, datasets
import spacy


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0, mask_fn=None):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.mask_fn = mask_fn
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad, self.mask_fn)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad, mask_fn):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & torch.tensor(
            mask_fn(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


class IWSLTIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)

        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def rebatch(pad_idx, batch):
    "Fix order in torchtext to match ours"
    src, trg = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
    return Batch(src, trg, pad_idx)


def get_datasets_and_vocab(dataset_path, cache=True):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text: str):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text: str):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<blank>"

    SRC = data.Field(tokenize=tokenize_de, pad_token=BLANK_WORD)
    TGT = data.Field(tokenize=tokenize_en, init_token=BOS_WORD,
                     eos_token=EOS_WORD, pad_token=BLANK_WORD)

    MAX_LEN = 100
    train, val, test = datasets.IWSLT.splits(
        exts=('.de', '.en'), fields=(SRC, TGT),
        root=dataset_path,
        filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(
            vars(x)['trg']) <= MAX_LEN
    )

    MIN_FREQ = 2

    SRC.build_vocab(train.src, min_freq=MIN_FREQ)
    TGT.build_vocab(train.trg, min_freq=MIN_FREQ)

    return SRC, TGT, train, val, test


class BatchConfig(object):
    _instance = None
    max_src_in_batch = 0
    max_tgt_in_batch = 0

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(BatchConfig, cls).__new__(
                cls
            )
        return cls._instance

    def __repr__(self) -> str:
        return f"Max src in batch = {self.max_src_in_batch} " +\
            f"Max tgt in batch = {self.max_tgt_in_batch}"


def batch_size_fn(new, count, sofar):
    "Keep augmenting batch and calculate total number of tokens + padding."
    bc = BatchConfig()
    if count == 1:
        bc.max_src_in_batch = 0
        bc.max_tgt_in_batch = 0
    bc.max_src_in_batch = max(bc.max_src_in_batch,  len(new.src))
    bc.max_tgt_in_batch = max(bc.max_tgt_in_batch,  len(new.trg) + 2)
    src_elements = count * bc.max_src_in_batch
    tgt_elements = count * bc.max_tgt_in_batch
    return max(src_elements, tgt_elements)
