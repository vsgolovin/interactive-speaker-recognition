from typing import Tuple, Iterable
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from kaldi_io import read_vec_flt_scp
from timit import read_spkrinfo
from common import PathLike


# helper functions
def read_vectors(scp_file: PathLike) -> Tuple[Tuple[str], Tuple[np.ndarray]]:
    "Read Kaldi vectors stored on disk"
    return zip(*read_vec_flt_scp(str(scp_file)))


def export_processed_embeddings(fname: PathLike, keys: Iterable[str],
                                embeddings: Iterable[np.ndarray]):
    data = dict(zip(keys, embeddings))
    np.savez(fname, **data)


def get_speaker_embeddings(X: np.ndarray, y: np.ndarray
                           ) -> Tuple[np.ndarray, np.ndarray]:
    "Average embeddings for every speaker"
    speakers = np.sort(np.unique(y))
    embeddings = np.zeros((len(speakers), X.shape[1]))
    for i, speaker in enumerate(speakers):
        mask = (y == speaker)
        embeddings[i] = np.mean(X[mask], axis=0)
    return embeddings, speakers


class XVectorPipeline:
    "Pipeline for processing raw x-vectors"
    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.speaker_embeddings = None
        self.mu = np.zeros(n_components)
        self.sigma = None

    def fit(self, X: np.ndarray, y: np.ndarray
            ) -> Tuple[np.ndarray, np.ndarray]:
        "Returns transformed embeddings for every speaker"
        # dimensionality reduction
        X_compressed = self.lda.fit_transform(X, y)

        # get speaker embeddings by averaging
        spkr_emb, speakers = get_speaker_embeddings(X_compressed, y)

        # normalize speaker embeddings
        self.mu = np.mean(spkr_emb, axis=0)
        spkr_emb -= self.mu
        norms = np.linalg.norm(spkr_emb, ord=2, axis=1)
        self.sigma = np.mean(norms)
        spkr_emb /= self.sigma

        return spkr_emb, speakers

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.sigma is not None, "Perform `fit` first"
        out = self.lda.transform(X)
        out -= self.mu
        return out / self.sigma


if __name__ == "__main__":
    from pathlib import Path

    def get_subset_dir(subset: str) -> Path:
        return Path(f"data/xvectors_{subset}")

    # map speaker ids to integers
    spkrinfo = read_spkrinfo("data/TIMIT/DOC/SPKRINFO.TXT")
    spkrs = np.array(spkrinfo.index)
    spkr2token = {spkr: i for i, spkr in enumerate(spkrs)}
    token2spkr = dict(enumerate(spkrs))

    pipeline = XVectorPipeline(n_components=128)

    # train subset (fit, transform and average embeddings)
    subset_dir = get_subset_dir("train")
    keys, embeddings = read_vectors(subset_dir / "xvector.scp")
    y = np.array([spkr2token[k.split("_")[0]] for k in keys])
    X = np.stack(embeddings)
    spk_embeddings, spk_y = pipeline.fit(X, y)
    export_processed_embeddings(subset_dir / "spk_xvector.npz",
                                [token2spkr[token] for token in spk_y],
                                embeddings)

    # test subset (transform and average)
    subset_dir = get_subset_dir("test")
    keys, embeddings = read_vectors(subset_dir / "xvector.scp")
    y = np.array([spkr2token[k.split("_")[0]] for k in keys])
    X = np.stack(embeddings)
    X_processed = pipeline.transform(X)
    embeddings, spk_y = get_speaker_embeddings(X_processed, y)
    export_processed_embeddings(subset_dir / "spk_xvector.npz",
                                [token2spkr[token] for token in spk_y],
                                embeddings)

    # test subset (transform)
    subset_dir = get_subset_dir("words")
    keys, embeddings = read_vectors(subset_dir / "xvector.scp")
    X_processed = pipeline.transform(np.stack(embeddings))
    export_processed_embeddings("data/xvectors_words/xvector.npz",
                                keys, X_processed)
